"""qwen 系列 LLM 输出 JSON 的逐级修复 + 容错解析钩子。

背景（评测日志 58 例真实失败分布）：检索链 4 个 LLM-JSON 解析点频繁解析失败
——'Expecting value' 15 例、"Expecting ',' delimiter" 13 例、'Unterminated'
（截断）若干、'Expecting property' 1 例。旧 ``utils.extract_json`` 的清理很弱：
盲替换 ``None``→``null`` 会误伤字符串内的 ``None`` 子串；只处理 ```json 围栏与
尾逗号；不处理 True/False/全角标点/截断。失败返 ``{}`` 即触发 llm_unavailable
降级路径（union 放行 / 拒答），实际损伤检索质量。

设计与手术边界
--------------
* **只服务检索链**：本模块只被 reasoning / super_tree / agentic.enhance /
  agentic.verifier 四处引用（均在索引缓存键清单 INDEX_CODE_FILES 的排除区内）。
  ``utils.py``（缓存键内、索引链也在用）保持原样不动 —— 索引行为零变化。
* **零改动直通**：能被 ``json.loads`` 直接解析的输入原样返回，不经过任何文本改写。
* **逐级最小改写**：修复按级别推进（见 REPAIR_STAGES），a–e 级为保守结构修复；
  截断补齐 (f) 与裸键/单引号 (g) 属激进级，只在"解析仍失败"时才升级应用，
  且管道在**第一个可解析的中间产物**处停止 —— 后续更激进的级别不再触碰它。
* **返回契约与旧 extract_json 一致**：成功返回 parsed 对象（dict/list/标量），
  失败返回 ``{}`` 并 logging.warning —— 下游降级分支语义不变。
* 每级修复函数独立导出、纯文本入出，便于单测断言"哪一级生效"。
  ``repair_json_text`` 返回 ``(repaired_text, applied_fixes)`` 供测试/日志观测。
"""

import json
import logging
import re

__all__ = [
    "repair_json_text",
    "extract_json_robust",
    "strip_code_fence",
    "clip_json_fragment",
    "escape_control_chars_in_strings",
    "normalize_fullwidth",
    "fix_python_literals",
    "remove_trailing_commas",
    "insert_missing_commas",
    "close_truncated",
    "quote_bare_keys",
    "convert_single_quotes",
    "REPAIR_STAGES",
    "ESCALATED_STAGES",
]

# ---------------------------------------------------------------------------
# 扫描内核：一次遍历得到 (in-string 标记, 字符串内容区间, 括号栈, 最外层闭合位置,
# 是否结束于未闭合字符串)。所有级别都基于它，保证"只在字符串字面量外改写"。
# ---------------------------------------------------------------------------

_OPEN2CLOSE = {"{": "}", "[": "]"}
_CLOSE2OPEN = {"}": "{", "]": "["}

# 掩盖串：字符串字面量**内容**逐字符替换为 'X'（引号本身保留，长度不变 →
# 掩码上的正则命中原子文本下标，可安全映射回原文做定点编辑）。
_MASK_CHAR = "X"


def _scan(text):
    """单遍结构扫描。

    返回 dict：
      flags  : list[bool]，len=len(text)；True 表示该字符处于字符串字面量内部
               （首尾引号本身为 False，使各级仍可改写定界符）
      spans  : list[(start, end)]，字符串**内容**的首/尾下标（含，未闭合时到文末）
      stack  : 扫描结束时仍未闭合的开括号栈（自外向内）
      closed_at : 首个"栈空"闭合符下标（-1 = 从未闭合，即截断/无 JSON）
      ends_in_string : 文本是否结束于未闭合字符串
    """
    n = len(text)
    flags = [False] * n
    spans = []
    stack = []
    closed_at = -1
    i = 0
    while i < n:
        ch = text[i]
        if ch == '"':
            start = i + 1
            i += 1
            closed = False
            while i < n:
                c = text[i]
                if c == "\\":
                    flags[i] = True
                    if i + 1 < n:
                        flags[i + 1] = True
                    i += 2
                    continue
                if c == '"':
                    closed = True
                    break
                flags[i] = True
                i += 1
            spans.append((start, i if closed else n))
            if closed:
                i += 1
                continue
            return {"flags": flags, "spans": spans, "stack": stack,
                    "closed_at": closed_at, "ends_in_string": True}
        if ch in _OPEN2CLOSE:
            stack.append(ch)
        elif ch in _CLOSE2OPEN:
            if stack and _OPEN2CLOSE[stack[-1]] == ch:
                stack.pop()
                if not stack and closed_at == -1:
                    closed_at = i
            elif not stack:
                pass  # 多余的闭括号：无法安全修复，原样留给解析失败
        i += 1
    return {"flags": flags, "spans": spans, "stack": stack,
            "closed_at": closed_at, "ends_in_string": False}


def _mask(text):
    """字符串内容替换为 'X' 的等长掩码（结构字符、引号、空白原样保留）。"""
    flags = _scan(text)["flags"]
    return "".join(_MASK_CHAR if f else c for f, c in zip(flags, text))


def _apply_edits(text, edits):
    """edits: [(start, end, replacement)]，坐标为**原文**下标，自动倒序应用。"""
    out = text
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + repl + out[end:]
    return out


# ---------------------------------------------------------------------------
# (a) 围栏剥离 + 最外层 JSON 片段夹取
# ---------------------------------------------------------------------------

def strip_code_fence(text):
    """剥离 ``` / ```json / ```text 围栏（含围栏前的说明性文字保留，交给夹取级）。

    无围栏时**原样返回**（不 strip，避免给 applied_fixes 记一条无意义级别）。
    """
    idx = text.find("```")
    if idx == -1:
        return text
    head = text[:idx]
    rest = text[idx + 3:]
    nl = rest.find("\n")
    body = rest[nl + 1:] if nl != -1 else rest.lstrip(" \t")
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    return (head + body).strip()


def clip_json_fragment(text):
    """取首个 ``{``/``[`` 到与之配对的 ``}``/``]``；无配对则取到文末（交给截断级）。

    找不到开括号时原样返回。
    """
    opens = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not opens:
        return text
    start = min(opens)
    tail = text[start:]
    info = _scan(tail)
    if info["closed_at"] != -1:
        clipped = tail[: info["closed_at"] + 1]
    else:
        clipped = tail
    # 只是裁掉首尾空白 → 原样返回（合法 JSON 直通零改写原则）
    if clipped == text.strip():
        return text
    return clipped


# ---------------------------------------------------------------------------
# (a.5) 字符串内的裸控制字符（真实换行/制表）→ 转义
# ---------------------------------------------------------------------------

_CTRL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def escape_control_chars_in_strings(text):
    """字符串字面量内的裸控制字符转义（json.loads 默认 strict 会直接报错）。

    旧 extract_json 用全局 ``\\n``→空格 掩盖该问题并顺带破坏正文语义；
    这里只在（已闭合的）字符串字面量内转义，长度变化但语义保真。
    """
    info = _scan(text)
    edits = []
    for start, end in info["spans"]:
        seg = text[start:end]
        if not any(ord(c) < 0x20 for c in seg):
            continue
        fixed = "".join(
            _CTRL_ESCAPES.get(c, "\\u%04x" % ord(c)) if ord(c) < 0x20 else c
            for c in seg
        )
        edits.append((start, end, fixed))
    return _apply_edits(text, edits) if edits else text


# ---------------------------------------------------------------------------
# (b) 全角标点归一（仅字符串外）
# ---------------------------------------------------------------------------

FULLWIDTH_MAP = {
    "，": ",",   # 全角逗号
    "：": ":",   # 全角冒号
    "（": "(",
    "）": ")",
    "\u201c": '"',  # 中文左双引号 “
    "\u201d": '"',  # 中文右双引号 ”
    "\u2018": '"',  # 中文左单引号 ‘
    "\u2019": '"',  # 中文右单引号 ’
    "｛": "{",
    "｝": "}",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
}


def normalize_fullwidth(text):
    """字符串外的全角标点 → ASCII 等价物。字符串内一字不动。"""
    flags = _scan(text)["flags"]
    out = list(text)
    changed = False
    for i, ch in enumerate(text):
        if flags[i]:
            continue
        rep = FULLWIDTH_MAP.get(ch)
        if rep:
            out[i] = rep
            changed = True
    return "".join(out) if changed else text


# ---------------------------------------------------------------------------
# (c) Python 字面量 → JSON 字面量（仅字符串外、词边界）
# ---------------------------------------------------------------------------

_PY_LITERAL_RE = re.compile(r"\b(True|False|None)\b")
_PY_LITERAL_MAP = {"True": "true", "False": "false", "None": "null"}


def fix_python_literals(text):
    """字符串外的 True/False/None → true/false/null。

    词边界 + in-string 屏蔽是关键：旧实现的盲 ``replace('None','null')`` 会把
    字符串值里的 ``NoneType`` 之类子串改坏（本函数不会）。
    """
    flags = _scan(text)["flags"]
    out = []
    i = 0
    n = len(text)
    changed = False
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append(text[i:j])
            i = j
            continue
        j = i
        while j < n and not flags[j]:
            j += 1
        seg = text[i:j]
        new = _PY_LITERAL_RE.sub(lambda m: _PY_LITERAL_MAP[m.group(0)], seg)
        changed |= new != seg
        out.append(new)
        i = j
    return "".join(out) if changed else text


# ---------------------------------------------------------------------------
# (d) 尾逗号：`,` 后紧跟 ]/} → 删逗号（仅字符串外）
# ---------------------------------------------------------------------------

_TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")


def remove_trailing_commas(text):
    mask = _mask(text)
    edits = [(m.start(), m.start() + 1, "") for m in _TRAILING_COMMA_RE.finditer(mask)]
    return _apply_edits(text, edits) if edits else text


# ---------------------------------------------------------------------------
# (e) 缺失逗号：值结束 + 新键/新元素开始之间只隔空白 → 插逗号
# ---------------------------------------------------------------------------

# 掩码上的 token 流（字符串已被压成 "XXXX"，故 "[X]*" 即一个完整字符串字面量）
_TOKEN_RE = re.compile(
    r'"[X]*"'
    r"|true|false|null"
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"|[{}\[\],:]"
    r"|[^\s{}\[\],:\"]+"
)
# 通配值 token（在掩码上重跑以判定类型）
_STRING_TOK = re.compile(r"^\"[X]*\"$")
_LITERAL_TOK = re.compile(r"^(?:true|false|null)$")
_NUMBER_TOK = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _token_kind(tok):
    if _STRING_TOK.match(tok):
        return "string"
    if _LITERAL_TOK.match(tok):
        return "literal"
    if _NUMBER_TOK.match(tok):
        return "number"
    if tok in _OPEN2CLOSE or tok in _CLOSE2OPEN:
        return tok
    return "other"


def insert_missing_commas(text):
    """值之后直接跟新键/新元素（无逗号）→ 插入逗号。

    判据（掩码 token 流上）：前一 token 是**值结束**（字符串/true/false/null/
    数字/``}``/``]``），紧邻的后一 token 是**值开始**（字符串/true/false/null/
    数字/``{``/``[``）。两 token 相邻即意味着它们之间没有任何 ``,``/``:``，
    而合法 JSON 中这种相邻不可能出现 → 插逗号是保语义的唯一解。
    （覆盖 ``["a" "b"]``、``{"a": true "b": false}``、``{"x": {"y":1} "z": 2}`` 等；
    包含题述保守子集。）
    """
    mask = _mask(text)
    prev = None
    prev_end = None
    edits = []
    for m in _TOKEN_RE.finditer(mask):
        kind = _token_kind(m.group(0))
        cur_start = m.start()
        if prev is not None and prev_end is not None:
            between = mask[prev_end:cur_start]
            if (
                between.strip() == ""
                and prev in ("string", "literal", "number", "}", "]")
                and kind in ("string", "literal", "number", "{", "[")
            ):
                edits.append((prev_end, prev_end, ","))
        prev = kind
        prev_end = m.end()
    return _apply_edits(text, edits) if edits else text


# ---------------------------------------------------------------------------
# (f) 截断补齐（Unterminated）
# ---------------------------------------------------------------------------

def close_truncated(text):
    """按开括号栈逆序补齐缺失的 ``]``/``}``；截断在字符串中间时先闭 ``"``。

    "停在边界"的截断另做回退，全部是**删除**残缺片段、不虚构值（宁可少一个键，
    也不给下游一个假的 true/null —— verifier 会把它当成真实判断消费掉）：
      * 尾随 ``,``              → 删（否则补齐后是 ``[1,2,]`` 非法）
      * 尾随 ``"key":``         → 连冒号与键一起回退删除
                                 （``{"a": 1, "b":`` → ``{"a": 1}``）
      * 对象内以悬空键结尾      → 回退删除（前一个 token 是 ``{``/``,``）；
                                 ``{"a": "x"`` 这类不动——冒号后的字符串是完整值
    """
    s = text.rstrip()
    if not s:
        return text

    info = _scan(s)
    if info["ends_in_string"]:
        # 末尾奇数个反斜杠会吃掉补上的引号 → 先补一个反斜杠闭合转义
        run = len(s) - len(s.rstrip("\\"))
        if run % 2 == 1:
            s += "\\"
        s += '"'
        info = _scan(s)

    if s.endswith(":"):
        cut = s[:-1].rstrip()
        cut_mask = _mask(cut)
        cut_toks = list(_TOKEN_RE.finditer(cut_mask))
        if cut_toks:
            last = cut_toks[-1]
            if _token_kind(cut_mask[last.start():last.end()]) in ("string", "other"):
                cut = cut[: last.start()].rstrip()
        if cut.endswith(","):
            cut = cut[:-1].rstrip()
        if cut:
            s = cut
            info = _scan(s)
    else:
        mask = _mask(s)
        toks = list(_TOKEN_RE.finditer(mask))
        if toks and info["stack"] and info["stack"][-1] == "{":
            last = toks[-1]
            prev = toks[-2] if len(toks) >= 2 else None
            prev_txt = mask[prev.start():prev.end()] if prev else None
            if (
                _token_kind(mask[last.start():last.end()]) == "string"
                and (prev_txt is None or prev_txt in ("{", ","))
            ):
                cut = s[: last.start()].rstrip()
                if cut.endswith(","):
                    cut = cut[:-1].rstrip()
                if cut:
                    s = cut
                    info = _scan(s)

    if s.endswith(","):
        s = s[:-1].rstrip()
        info = _scan(s)

    closers = "".join(_OPEN2CLOSE[ch] for ch in reversed(info["stack"]))
    if not closers:
        return text
    return s + closers


# ---------------------------------------------------------------------------
# (g) 裸键加引号 / 单引号字符串 → 双引号（保守子集）
# ---------------------------------------------------------------------------

_BARE_KEY_RE = re.compile(r"(?<=[{,])(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)(?=\s*:)")


def quote_bare_keys(text):
    """裸键 ``{a: 1}`` / ``{..., b: 2}`` → ``{"a": 1}``/``{..., "b": 2}``（字符串外）。"""
    mask = _mask(text)
    edits = [
        (m.start(2), m.end(2), '"' + text[m.start(2):m.end(2)] + '"')
        for m in _BARE_KEY_RE.finditer(mask)
    ]
    return _apply_edits(text, edits) if edits else text


def convert_single_quotes(text):
    """单引号字符串 ``'...'`` → 双引号。**只在**内部不含 ``"`` 时改写（保守）。

    含双引号的单引号串需要转义 ``\\"``，容易误判 → 直接跳过（宁可返回失败让
    下游降级，不可改坏 JSON 语义）。
    """
    mask = _mask(text)
    edits = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i] != "'":
            i += 1
            continue
        j = i + 1
        ok = False
        while j < n:
            c = mask[j]
            if c == "\\":
                j += 2
                continue
            if c == "'":
                ok = True
                break
            if c == '"':
                break
            if c == "\n":
                break
            j += 1
        if not ok:
            i += 1
            continue
        inner = text[i + 1:j]
        if '"' not in inner:
            # JSON 不允许 \' 转义 → 单引号串内的 \' 还原为裸 '
            if "\\'" in inner:
                edits.append((i + 1, j, inner.replace("\\'", "'")))
            edits.append((i, i + 1, '"'))
            edits.append((j, j + 1, '"'))
        i = j + 1
    return _apply_edits(text, edits) if edits else text


# ---------------------------------------------------------------------------
# 管道
# ---------------------------------------------------------------------------

#: 保守级（无条件依次应用）：围栏剥离 → 夹取 → 控制字符 → 全角归一 →
#: Python 字面量 → 尾逗号 → 缺失逗号
REPAIR_STAGES = (
    ("strip_fence", strip_code_fence),
    ("clip_fragment", clip_json_fragment),
    ("control_chars", escape_control_chars_in_strings),
    ("fullwidth", normalize_fullwidth),
    ("python_literals", fix_python_literals),
    ("trailing_commas", remove_trailing_commas),
    ("missing_commas", insert_missing_commas),
)

#: 激进级（仅在前级产物仍不可解析时逐级升级）：截断补齐 → 裸键 → 单引号
ESCALATED_STAGES = (
    ("close_truncated", close_truncated),
    ("bare_keys", quote_bare_keys),
    ("single_quotes", convert_single_quotes),
)


def _loads(text):
    try:
        return json.loads(text), None
    except Exception as e:  # JSONDecodeError 及其它防御
        return None, e


def repair_json_text(text, stop_on_success=True):
    """逐级修复，返回 ``(repaired_text, applied_fixes)``。

    * ``applied_fixes``：生效级别名列表（顺序即应用顺序），测试/日志据此断言"哪级救的"。
    * ``stop_on_success=True``（默认）：保守级产物一旦可解析就停止，激进级
      （截断补齐 / 裸键 / 单引号）不再触碰 —— 最小改写原则。
    * 本函数**不负责解析**：返回值仍是文本，是否能解析由调用方判定。
    """
    fixes = []
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else "", fixes

    cur = text
    for name, fn in REPAIR_STAGES:
        try:
            new = fn(cur)
        except Exception as e:  # 任何一级内部异常都不允许冒到检索链
            logging.debug("json_repair stage %s skipped: %s", name, e)
            continue
        if new != cur:
            cur = new
            fixes.append(name)

    if stop_on_success and _loads(cur)[0] is not None:
        return cur, fixes

    for name, fn in ESCALATED_STAGES:
        try:
            new = fn(cur)
        except Exception as e:
            logging.debug("json_repair stage %s skipped: %s", name, e)
            continue
        if new != cur:
            cur = new
            fixes.append(name)
        if stop_on_success and _loads(cur)[0] is not None:
            break
    return cur, fixes


def _coarse_slice(text):
    """现有 extract_json 的兜底思路：首 ``{``/``[`` 到 尾 ``}``/``]`` 的粗夹取。"""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if not starts or not ends:
        return None
    s, e = min(starts), max(ends)
    if e <= s:
        return None
    return text[s:e + 1]


def extract_json_robust(content):
    """检索链 LLM-JSON 解析钩子（旧 ``extract_json`` 的加固替代）。

    步骤：直接 ``json.loads``（**零改动直通**）→ 失败则 ``repair_json_text`` 全流程
    → 再 loads → 失败则首/尾括号粗夹取 + 再修复 + 再 loads → 仍失败返 ``{}``
    （与旧契约一致，下游 llm_unavailable 降级语义不变）。
    """
    if content is None:
        return {}
    if not isinstance(content, str):
        logging.warning("extract_json_robust: 非字符串输入 %r", type(content).__name__)
        return {}
    if not content.strip():
        return {}

    parsed, err = _loads(content)
    if err is None:
        return parsed

    repaired, fixes = repair_json_text(content)
    if repaired != content:
        parsed, err = _loads(repaired)
        if err is None:
            logging.info("extract_json_robust: 修复成功 levels=%s", fixes)
            return parsed

    coarse = _coarse_slice(repaired if repaired.strip() else content)
    if coarse:
        repaired2, fixes2 = repair_json_text(coarse)
        parsed, err = _loads(repaired2)
        if err is None:
            logging.info("extract_json_robust: 修复成功 levels=%s+coarse", fixes + fixes2)
            return parsed

    logging.warning(
        "extract_json_robust: JSON 解析失败 (%s)，已尝试修复 levels=%s，返回 {}。原文前 200 字: %r",
        err, fixes, content[:200],
    )
    return {}
