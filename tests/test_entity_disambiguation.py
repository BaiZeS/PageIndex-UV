"""P1.1 实体消歧增强测试 —— blocking 预裁剪 + 批归一分块 map-reduce + 类型分期。

验收覆盖：
1. blocking 轻量信号（纯函数，无 LLM）：精确/前缀/Jaccard/编辑距离；
   "小张" vs "张三" 产生疑似对（编辑距离=2 且共享 "张"），明显不匹配对不通过；
   非字符串脏条目被过滤而非抛错；
2. disambiguate_entity：无幸存候选 → 跳过 LLM；有 → LLM 只见幸存者（top-N 上限）；
   候选别名含非字符串条目不崩且字符串别名仍生效；
3. 批归一分块：>200 名字分块 map + 有界迭代分块 reduce（收敛守卫：
   代表元 ≤ 块大小单次收口 / 恒等映射不动点 / 最大轮数）；单次 prompt 永不超块大小；
   单块 LLM 失败 → 该块恒等，其余不受影响；别名累积到 canonical；
4. 类型分期：concept 不进入主动批归一（person/project/organization 覆盖）。

全部 LLM 调用均 mock —— 无真实 LLM、无向量（FULLY VECTORLESS）。
"""
import json
import logging
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB

# NOTE（测试隔离守卫，与 test_retrieve_model_wiring.py 同理）：test_corpus_tree /
# test_router 等文件会在运行期清空并重建 sys.modules 中的 pageindex_mutil.* 模块
# 对象，字符串路径 patch 可能落到与被测类引用不同的模块上。此处导入时清理可能被
# 预置的 stub 并持有真实模块对象引用，patch 一律用 patch.object(module, ...)，
# 保证命中被测类实际引用的模块。
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]

import pageindex_mutil.entity_extractor as entity_extractor_mod
from pageindex_mutil.entity_extractor import (
    EntityExtractor,
    _blocking_score,
    _edit_distance,
    NORMALIZE_BATCH_CHUNK_SIZE,
    NORMALIZE_REDUCE_MAX_ROUNDS,
    BATCH_NORMALIZE_ENTITY_TYPES,
    BLOCKING_EXACT_SCORE,
)


@pytest.fixture
def db(tmp_path):
    """Create a fresh temp DB for each test."""
    db_path = str(tmp_path / "test.db")
    db = PageIndexDB(db_path)
    yield db
    db.close()


# ===========================================================================
# 1. blocking 轻量信号（纯函数，无 LLM）
# ===========================================================================

class TestBlockingPredicate:
    """blocking 只做 LLM 前置裁剪：任一信号命中 → 疑似对；否则 None。"""

    def test_short_cjk_shared_char_suspected(self):
        """"小张" vs "张三"：编辑距离=2≤阈值 且共享 "张" → 疑似对（保守但连通）。"""
        score = _blocking_score(["小张"], ["张三"])
        assert score is not None, "小张/张三 应产生疑似对（交 LLM 裁定，不直接合并）"
        assert score > 0

    def test_obvious_mismatch_rejected(self):
        """明显不匹配："帮会系统" vs "浴血值" → 无信号。"""
        assert _blocking_score(["帮会系统"], ["浴血值"]) is None

    def test_no_shared_char_short_pair_rejected(self):
        """短名字编辑距离=2 但零字符交集（整串替换）→ 不作为相似信号。"""
        assert _blocking_score(["小明"], ["阿红"]) is None

    def test_single_char_non_prefix_rejected(self):
        """单字非前缀且距离超阈值 → 不通过。"""
        assert _blocking_score(["会"], ["帮会系统"]) is None

    def test_exact_match_case_insensitive(self):
        """精确匹配（case-insensitive）→ 满分 1.0。"""
        assert _blocking_score(["zhang san"], ["Zhang San"]) == 1.0
        assert _blocking_score(["  张三 "], ["张三"]) == 1.0

    def test_prefix_match_passes(self):
        """前缀匹配（≥2 字）→ 疑似对。"""
        score = _blocking_score(["风控"], ["风控系统"])
        assert score is not None
        assert score >= 0.9
        # 反向亦然：新实体是长名，候选是短前缀
        assert _blocking_score(["张三丰"], ["张三"]) is not None

    def test_jaccard_char_overlap_passes(self):
        """字符集高度重叠（Jaccard ≥ 阈值）→ 疑似对。"""
        assert _blocking_score(["数据安全"], ["安全数据"]) is not None

    def test_edit_distance_scales_for_long_names(self):
        """长名字阈值放宽：12 字距离 3 = max(2, 12//4) 通过；距离 8 不通过。"""
        base = "abcdefghijkl"
        near = "abcxyzghijkl"  # 3 处替换
        far = "xyzwefghopqr"  # 8 处替换，Jaccard=0.2 < 0.5，无前缀
        assert _edit_distance(base, near) == 3
        assert _blocking_score([base], [near]) is not None
        assert _edit_distance(base, far) == 8
        assert _blocking_score([base], [far]) is None

    def test_candidate_aliases_considered(self):
        """候选别名参与 blocking：新名 vs 候选别名前缀命中。"""
        assert _blocking_score(["张三丰"], ["张三", "老张"]) is not None

    def test_query_aliases_considered(self):
        """新实体别名参与 blocking。"""
        assert _blocking_score(["张先生", "老张"], ["张三"]) is not None

    def test_empty_inputs_rejected(self):
        """空输入 → None。"""
        assert _blocking_score([], ["张三"]) is None
        assert _blocking_score(["张三"], []) is None
        assert _blocking_score(["", "  "], ["张三"]) is None

    def test_non_string_entries_filtered_not_crash(self):
        """非字符串脏条目（如 int 别名）被过滤而非 AttributeError；字符串条目仍生效。"""
        assert _blocking_score(["张三"], [123]) is None
        assert _blocking_score([123], ["张三"]) is None
        assert _blocking_score(["小张三"], [123, "小张三"]) == BLOCKING_EXACT_SCORE

    def test_length_diff_prune_keeps_behavior(self):
        """长度差超编辑距离阈值 → 免跑 DP 直接判否（与原距离判定结果一致）。"""
        assert _edit_distance("张", "张三丰系统") == 4  # 距离 4 > 阈值 max(2, 5//4)=2
        assert _blocking_score(["张"], ["张三丰系统"]) is None


# ===========================================================================
# 2. disambiguate_entity 集成 blocking
# ===========================================================================

class TestDisambiguateEntityWithBlocking:
    """无幸存候选跳过 LLM；有则 LLM 只见幸存者。"""

    def test_no_suspects_skips_llm(self):
        """blocking 无幸存候选 → 不调 LLM，保守不合并。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        existing = [{"id": 1, "name": "帮会系统", "aliases": "[]", "entity_type": "concept"}]
        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            result = extractor.disambiguate_entity("浴血值", [], existing)
            assert result is None
            mock_llm.assert_not_called()

    def test_llm_sees_only_survivors(self):
        """LLM prompt 只包含 blocking 幸存者，不含未命中候选。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        existing = [
            {"id": 1, "name": "张三", "aliases": "[]", "entity_type": "person"},
            {"id": 2, "name": "浴血值", "aliases": "[]", "entity_type": "person"},
            {"id": 3, "name": "数据安全", "aliases": "[]", "entity_type": "person"},
        ]
        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "should_merge": True, "canonical_name": "张三", "reason": "同一人"
            })
            result = extractor.disambiguate_entity("张先生", [], existing)
            assert result is not None and result["id"] == 1
            mock_llm.assert_called_once()
            prompt = mock_llm.call_args[0][1]
            assert "张三" in prompt
            assert "浴血值" not in prompt, "未命中 blocking 的候选不应进入 LLM prompt"
            assert "数据安全" not in prompt
            # NFR4：使用 retrieve_model
            assert mock_llm.call_args[0][0] == "r-model"

    def test_candidate_cap_top_n(self):
        """幸存者过多时只取 top-N（BLOCKING_MAX_CANDIDATES=20）。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        existing = [
            {"id": i, "name": f"候选者{i:02d}", "aliases": "[]", "entity_type": "person"}
            for i in range(25)
        ]
        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({"should_merge": False})
            extractor.disambiguate_entity("候选者", [], existing)
            prompt = mock_llm.call_args[0][1]
            # 稳定排序保留前 20 个
            assert "候选者19" in prompt
            assert "候选者20" not in prompt, "超过 top-20 的幸存候选应被裁剪"

    def test_llm_failure_still_conservative(self):
        """有幸存候选但 LLM 失败 → 保守不合并。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        existing = [{"id": 1, "name": "张三", "aliases": "[]", "entity_type": "person"}]
        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = None
            assert extractor.disambiguate_entity("张先生", [], existing) is None

    def test_entity_row_non_string_alias_no_crash(self):
        """候选实体行别名 JSON 含非字符串条目：blocking 不崩且字符串别名仍生效。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        existing = [
            {"id": 1, "name": "王五", "aliases": json.dumps([123, "小张三"]),
             "entity_type": "person"},
        ]
        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "should_merge": True, "canonical_name": "王五", "reason": "别名命中"
            })
            # 修复前：123.strip() → AttributeError
            result = extractor.disambiguate_entity("小张三", [], existing)
            assert result is not None and result["id"] == 1
            # 字符串别名 "小张三" 精确命中 blocking → 进入 LLM 裁定
            mock_llm.assert_called_once()


# ===========================================================================
# 2b. extract_entities 别名清洗
# ===========================================================================

class TestExtractEntitiesAliasSanitization:
    """别名入口清洗：只保留非空字符串，非 str 条目直接丢弃（不强转）。"""

    def test_non_string_aliases_dropped(self):
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        payload = json.dumps([
            {"name": "张三", "type": "person",
             "aliases": [123, "小张三", "", "   ", None], "confidence": 0.9},
        ], ensure_ascii=False)
        with patch.object(entity_extractor_mod, "llm_completion",
                          return_value=payload):
            entities = extractor.extract_entities("doc.pdf", "测试文档", ["章节一"])
        assert len(entities) == 1
        assert entities[0].aliases == ["小张三"]


# ===========================================================================
# 3. 批归一分块 map-reduce
# ===========================================================================

class TestNormalizeChunking:
    """按 NORMALIZE_BATCH_CHUNK_SIZE 分块 map；reduce 有界迭代分块归并：
    代表元 ≤ 块大小 → 单次收口；恒等映射 → 不动点停；最多 NORMALIZE_REDUCE_MAX_ROUNDS 轮。"""

    def test_single_chunk_merge_and_alias_accumulation(self, db):
        """≤200 名字：单次 LLM 调用；合并后别名累积且可通过 search_entities 命中。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        db.insert_entity("person", "张三")
        db.insert_entity("person", "小张")
        db.insert_entity("person", "李四")

        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "groups": [{"canonical": "张三", "synonyms": ["张三", "小张"]}]
            })
            extractor.normalize_entities_batch(db)
            assert mock_llm.call_count == 1

        entities = db.get_entities_by_type("person")
        names = {e["name"] for e in entities}
        assert names == {"张三", "李四"}, "小张 应并入 张三"
        zhang_san = db.get_entity_by_name("张三")
        aliases = json.loads(zhang_san.get("aliases", "[]"))
        assert "小张" in aliases, "合并后被并入实体的名字应累积为 canonical 别名"
        hits = {r["name"] for r in db.search_entities("小张")}
        assert "张三" in hits, "小张 作为别名应可检索到 张三"

    def test_large_corpus_chunked_with_consolidation(self, db):
        """>200 名字：map/reduce 每次调用 ≤200 名字；零合并的 reduce 轮经不动点终止。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        total = NORMALIZE_BATCH_CHUNK_SIZE * 2 + 50  # 450 → 3 map 块 + reduce 轮 3 块
        for i in range(total):
            db.insert_entity("person", f"人员{i:03d}")

        calls = []

        def fake_llm(model, prompt, **kw):
            calls.append(prompt)
            assert model == "r-model", "NFR4: 批归一应使用 retrieve_model"
            if "人员001" in prompt:
                # 仅 map 第一块同时含 人员000/人员001（reduce 轮已无 人员001）
                return json.dumps({
                    "groups": [{"canonical": "人员000", "synonyms": ["人员000", "人员001"]}]
                })
            return json.dumps({"groups": []})

        with patch.object(entity_extractor_mod, "llm_completion", side_effect=fake_llm):
            extractor.normalize_entities_batch(db)

        # 3 map 块 + reduce 第 1 轮：449 代表元 → 3 块（200/200/49），全零合并 → 不动点停
        assert len(calls) == 6
        for i, prompt in enumerate(calls):
            count = prompt.count("人员")
            assert count <= NORMALIZE_BATCH_CHUNK_SIZE, f"调用 {i} 名字数超限: {count}"
        assert calls[0].count("人员") == NORMALIZE_BATCH_CHUNK_SIZE
        assert calls[2].count("人员") == 50
        # reduce 轮输入为去重代表元：含合并后的 人员000 与块 3 的 人员449，不再有 人员001
        reduce_calls = calls[3:]
        assert "人员000" in reduce_calls[0]
        assert "人员449" in reduce_calls[-1]
        assert all("人员001" not in p for p in reduce_calls), \
            "人员001 已在块内并入，不应再作为代表元"

        # 合并应用：人员001 删除，别名累积到 人员000
        entities = db.get_entities_by_type("person")
        assert len(entities) == total - 1
        assert db.get_entity_by_name("人员001") is None
        canonical = db.get_entity_by_name("人员000")
        aliases = json.loads(canonical.get("aliases", "[]"))
        assert "人员001" in aliases

    def test_exactly_chunk_size_no_consolidation(self, db):
        """恰好 200 名字：单块单次调用，无 reduce 轮。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        for i in range(NORMALIZE_BATCH_CHUNK_SIZE):
            db.insert_entity("person", f"边界{i:03d}")
        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({"groups": []})
            extractor.normalize_entities_batch(db)
            assert mock_llm.call_count == 1, "单块不应触发 reduce 轮"

    def test_reduce_merges_cross_chunk_representatives(self, db):
        """跨块同义：块内未见全量，map 后代表元 ≤ 块大小 → 最终单次收口合并跨块对。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        db.insert_entity("person", "张三")
        db.insert_entity("person", "张先生")
        for i in range(198):  # 填满第一块（200 个）
            db.insert_entity("person", f"填充{i:03d}")
        db.insert_entity("person", "张总")  # 第 201 个 → 独立第二块

        calls = []

        def fake_llm(model, prompt, **kw):
            calls.append(prompt)
            if "张总" in prompt:
                # reduce 最终收口：同时见到 张总（第二块代表元）与第一块代表元
                return json.dumps({
                    "groups": [{"canonical": "张三", "synonyms": ["张三", "张总"]}]
                })
            if prompt.count("张先生") >= 2:
                # map 块 1（prompt 模板示例文本自带 1 个 "张先生"，名单中再含 1 个 → ≥2）
                return json.dumps({
                    "groups": [{"canonical": "张三", "synonyms": ["张三", "张先生"]}]
                })
            return json.dumps({"groups": []})

        with patch.object(entity_extractor_mod, "llm_completion", side_effect=fake_llm):
            extractor.normalize_entities_batch(db)

        # map 块 1（200 名；第二块仅 张总 1 个名字，不调 LLM）：张先生→张三，
        # 代表元降为 200（≤ 块大小）→ reduce 走最终单次收口（分支 a），合并 张总
        assert len(calls) == 2
        assert calls[1].count("张总") == 1
        assert db.get_entity_by_name("张总") is None
        assert db.get_entity_by_name("张先生") is None
        canonical = db.get_entity_by_name("张三")
        aliases = json.loads(canonical.get("aliases", "[]"))
        assert "张总" in aliases, "reduce 收口跨块合并后别名应累积"
        assert "张先生" in aliases

    def test_zero_merge_reduce_bounded_and_stops(self, db):
        """零合并 600 名：每次调用 ≤200 名字；reduce 第 1 轮即因"无变化"终止，不死循环。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        total = NORMALIZE_BATCH_CHUNK_SIZE * 3  # 600
        for i in range(total):
            db.insert_entity("person", f"成员{i:04d}")

        calls = []

        def fake_llm(model, prompt, **kw):
            calls.append(prompt)
            return json.dumps({"groups": []})  # 全程零合并

        with patch.object(entity_extractor_mod, "llm_completion", side_effect=fake_llm):
            extractor.normalize_entities_batch(db)

        # 3 map 块 + reduce 第 1 轮（600 代表元 → 3 块），恒等映射 → 不动点停；
        # 若无收敛守卫会继续跑满 NORMALIZE_REDUCE_MAX_ROUNDS 轮（共 12 次调用）
        assert len(calls) == 6
        for i, prompt in enumerate(calls):
            count = prompt.count("成员")
            assert 0 < count <= NORMALIZE_BATCH_CHUNK_SIZE, \
                f"调用 {i} 名字数超限: {count}"
        # 无合并：600 个实体全部保留
        assert len(db.get_entities_by_type("person")) == total

    def test_reduce_max_rounds_warns_and_keeps_mapping(self, db, caplog):
        """每轮都有合并但代表元始终 > 块大小 → 达 NORMALIZE_REDUCE_MAX_ROUNDS 轮终止并告警。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        total = NORMALIZE_BATCH_CHUNK_SIZE * 3  # 600
        for i in range(total):
            db.insert_entity("person", f"名目{i:04d}")

        call_lists = []

        def fake_llm(model, prompt, **kw):
            names = json.loads(re.search(r"\[[^\]]*\]", prompt).group(0))
            call_lists.append(names)
            if len(names) >= 2:
                # 每块恒合并头两个名字 → 每轮有变化但代表元仅每块减 1，始终 > 块大小
                return json.dumps({
                    "groups": [{"canonical": names[0], "synonyms": [names[0], names[1]]}]
                })
            return json.dumps({"groups": []})

        with patch.object(entity_extractor_mod, "llm_completion",
                          side_effect=fake_llm):
            with caplog.at_level(logging.WARNING):
                extractor.normalize_entities_batch(db)

        # map 3 块 + NORMALIZE_REDUCE_MAX_ROUNDS 轮 × 3 块 = 12 次调用，按轮数上限终止
        assert len(call_lists) == 3 + 3 * NORMALIZE_REDUCE_MAX_ROUNDS
        assert all(len(names) <= NORMALIZE_BATCH_CHUNK_SIZE for names in call_lists)
        assert any("did not converge" in r.message for r in caplog.records), \
            "达到最大 reduce 轮数应记 warning"
        # 已发生的合并仍被应用（保留当前映射，不丢弃）
        assert db.get_entity_by_name("名目0001") is None
        aliases = json.loads(db.get_entity_by_name("名目0000").get("aliases", "[]"))
        assert "名目0001" in aliases

    def test_one_chunk_llm_failure_maps_identity_only_that_chunk(self, db):
        """多块中一块 LLM 返回 None → 该块恒等映射，其他块与 reduce 不受影响。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        total = NORMALIZE_BATCH_CHUNK_SIZE * 2 + 50  # 450 → 3 map 块
        for i in range(total):
            db.insert_entity("person", f"职员{i:03d}")

        calls = []
        state = {"n": 0}

        def fake_llm(model, prompt, **kw):
            state["n"] += 1
            calls.append(prompt)
            if state["n"] == 2:
                return None  # map 块 2 LLM 失败 → 保守恒等
            if "职员001" in prompt:
                return json.dumps({
                    "groups": [{"canonical": "职员000", "synonyms": ["职员000", "职员001"]}]
                })
            return json.dumps({"groups": []})

        with patch.object(entity_extractor_mod, "llm_completion", side_effect=fake_llm):
            extractor.normalize_entities_batch(db)

        # 3 map 块 + reduce 轮 3 块（449 代表元，全零合并 → 不动点停）
        assert len(calls) == 6
        # 块 1 合并正常应用
        assert db.get_entity_by_name("职员001") is None
        aliases = json.loads(db.get_entity_by_name("职员000").get("aliases", "[]"))
        assert "职员001" in aliases
        # 失败块 2 恒等：200 个实体全数保留
        for i in range(NORMALIZE_BATCH_CHUNK_SIZE, NORMALIZE_BATCH_CHUNK_SIZE * 2):
            assert db.get_entity_by_name(f"职员{i:03d}") is not None
        assert len(db.get_entities_by_type("person")) == total - 1


# ===========================================================================
# 4. 类型分期
# ===========================================================================

class TestTypePhasing:
    """主动批归一只覆盖 person/project/organization，concept 后置。"""

    def test_phased_types_constant(self):
        assert set(BATCH_NORMALIZE_ENTITY_TYPES) == {"person", "project", "organization"}
        assert "concept" not in BATCH_NORMALIZE_ENTITY_TYPES

    def test_concept_never_fed_to_batch_normalization(self, db):
        """concept 名字不进入任何批归一 prompt；person/project 正常归一。"""
        extractor = EntityExtractor(model="m", retrieve_model="r-model")
        db.insert_entity("person", "张三")
        db.insert_entity("person", "张先生")
        db.insert_entity("project", "风控平台")
        db.insert_entity("project", "风控系统平台")
        db.insert_entity("concept", "数据安全")
        db.insert_entity("concept", "信息安全")

        with patch.object(entity_extractor_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({"groups": []})
            extractor.normalize_entities_batch(db)

        assert mock_llm.call_count == 2, "person + project 各一轮，concept 跳过"
        prompts = [c[0][1] for c in mock_llm.call_args_list]
        assert all("数据安全" not in p and "信息安全" not in p for p in prompts)
        # concept 实体保持原样（未合并、未删除）
        concept_names = {e["name"] for e in db.get_entities_by_type("concept")}
        assert concept_names == {"数据安全", "信息安全"}
