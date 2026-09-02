"""closet 语义标签（Channel A）查询期检索——子串语义，与渲染侧 `_tag_hits` 同一标准。

缺陷与修复缘由（A0/geneal-6 插桩实证，mldr_zh#0 bundle: kw=40 ent=113 **tag=0**）
------------------------------------------------------------------------
* 索引期 `closet_index._tokenize_tag`（**保持不变，属缓存键排除区**）把 `tag_token`
  存成**空格连接的多词串**——抽样 1451 条 97.9% 为多 token，如
  `'游戏 世界观 门派 体系'`。这是设计形状，不是脏数据。
* 旧查询期链路 `ClosetIndex.search` → `db.match_closet_tags` 用
  `SELECT ... WHERE tag_token IN (?, ?, ...)` 拿**单个查询词整串精确匹配**多词串，
  真实标签上几乎不可能等值 → **查询期恒零命中**。
* 渲染侧 `agentic/enhance.py::UnifiedNodeEnhancement._tag_hits`（enhance.py:121-137）
  一直是**子串**语义且能命中。搜索侧与渲染侧标准不一致即本缺陷本质。

修法：**只动查询侧适配存储形状**（标签存储不动、索引链 9 个 INDEX_CODE_FILES 零改动）。
本模块一次 SELECT 取本 workspace 全部 `source='llm'` 标签行（百量级），在 Python 侧按
`_tag_hits` 同一套子串规则判定命中，`score = 命中 tag 的 confidence 之和`
（沿 `match_closet_tags` 旧契约），按 doc 聚合排序。

性能红线：**每查询一次 SELECT**，不做逐 doc 回填查询（旧路径 matched_docs ×
`get_doc_tags` 的 N+1 在此消解——命中标签文本随扫描一次带出）。
"""
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _norm_terms(terms: Optional[Iterable[str]]) -> List[str]:
    """词项归一：strip + casefold + 去重（保序）。子串匹配大小写不敏感。"""
    out: List[str] = []
    seen = set()
    for t in terms or []:
        if not isinstance(t, str):
            continue
        cf = t.strip().casefold()
        if cf and cf not in seen:
            seen.add(cf)
            out.append(cf)
    return out


def _candidate_hit(cand_cf: str, tokens_cf: Sequence[str], query_cf: str) -> bool:
    """单个候选串（tag_text / tag_token）对 query 的命中判定。

    **逐条镜像** `UnifiedNodeEnhancement._tag_hits`（enhance.py:121-137）的规则，
    使搜索侧与渲染侧标准严格一致：
      1. 候选串整体出现于 query（`len >= 2` 且 `cand in query_cf`）→ 命中；
      2. 任一 query 词项与候选串等值，或两者互为学生串（**两侧均 `len >= 2`** 才判子串）→ 命中。
    """
    if not cand_cf:
        return False
    if len(cand_cf) >= 2 and cand_cf in query_cf:
        return True
    for qt in tokens_cf:
        if qt == cand_cf:
            return True
        if len(qt) >= 2 and len(cand_cf) >= 2 and (qt in cand_cf or cand_cf in qt):
            return True
    return False


def row_hits(tag_text: str, tag_token: str, tokens_cf: Sequence[str],
             query_cf: str) -> bool:
    """标签行命中判定：`tag_text` 或 `tag_token`（多词串）任一按 `_tag_hits` 规则命中。

    tag_text 维与渲染侧完全同标准；tag_token 维额外覆盖「查询词项只匹配到标签的
    某个分词片段」的情形（多词串本就是 tag_text 的分词，故为召回补集而非放宽）。
    """
    if _candidate_hit((tag_text or "").strip().casefold(), tokens_cf, query_cf):
        return True
    return _candidate_hit((tag_token or "").strip().casefold(), tokens_cf, query_cf)


def search_tags(client=None, db=None, query_tokens: Optional[Sequence[str]] = None,
                topk: int = 30, tokens_for_substring: Optional[Sequence[str]] = None,
                query: str = "") -> List[Tuple[int, float, List[Tuple[str, float]]]]:
    """closet `source='llm'` 语义标签通道的子串检索。

    Args:
        client: `PageIndexClient`；`db` 缺省时回退 `client.db`。
        db: `PageIndexDB`（走其既有线程本地连接 `_connect()`，与 evidence.py 关键词
            provenance 查询同法；**不新增 db 接口、不改 db.py**）。
        query_tokens: 查询词项（旧 `match_closet_tags` 契约的输入）。调用方传
            `KeywordIndex._tokenize(None, query)` 的结果——与证据束 ctx["tokens"]、
            渲染侧 `_tag_hits` 同一份词项，天然两侧一致。
        topk: 返回文档数上限。
        tokens_for_substring: 子串判定实际使用的词项；缺省复用 `query_tokens`
            （调用方两处同一份，保留形参仅为口径可覆盖）。
        query: 原始查询串，供规则 1（标签整体出现于 query）；缺省由词项拼接。

    Returns:
        `[(doc_id, score, [(tag_text, confidence), ...]), ...]`——score 为该 doc
        命中标签 confidence 之和，按 score 降序（同分按 doc_id 升序，确定性）取前 topk；
        第三元为命中标签明细（provenance），未命中任何行时为空列表。
    """
    tokens_cf = _norm_terms(
        query_tokens if tokens_for_substring is None else tokens_for_substring)
    if not tokens_cf:
        return []
    db_obj = db if db is not None else getattr(client, "db", None)
    if db_obj is None:
        logger.debug("tag_search: no db handle, skipping tag channel")
        return []
    query_cf = (query or "").casefold() or " ".join(tokens_cf)

    # 唯一一次 SELECT：本 workspace 全部 llm 标签行（每文档 ≤5 条，百量级）。
    rows = db_obj._connect().execute(
        "SELECT doc_id, tag_text, tag_token, confidence FROM closet_tags "
        "WHERE source = 'llm'").fetchall()

    agg: dict = {}
    for r in rows:
        doc_id, tag_text, tag_token, confidence = r[0], r[1], r[2], r[3]
        if not row_hits(tag_text, tag_token, tokens_cf, query_cf):
            continue
        try:
            conf = float(confidence or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        did = int(doc_id)
        if did not in agg:
            agg[did] = [0.0, []]
        agg[did][0] += conf
        agg[did][1].append(((tag_text or "").strip(), conf))

    ranked = sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))[:topk]
    return [(did, score, hits) for did, (score, hits) in ranked]
