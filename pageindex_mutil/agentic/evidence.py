"""证据束：L0 四通道原始命中 + 图谱关联的 query 级缓存对象（spec [S5]）。"""
import logging


def _dedup(items, key):
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def derive_evidence_score(entry) -> float:
    """文档级证据分：3*实体 + 2*标签 + 1*关键词（仅用于分组序/补充清单序/matched_docs，不参与裁定）。"""
    ch = (entry or {}).get("channels") or {}
    return (
        3.0 * len(ch.get("entity") or [])
        + 2.0 * len(ch.get("tag") or [])
        + 1.0 * len(ch.get("keyword") or [])
    )


def build_evidence_bundle(client, db, query, topk=30) -> dict:
    """构建证据束。返回 {db_id: {"channels": {...}, "graph": {...}}}。"""
    bundle = {}

    def entry(db_id):
        return bundle.setdefault(int(db_id), {
            "channels": {"tag": [], "keyword": [], "entity": [], "vector": []},
            "graph": {"doc_entity_links": []},
        })

    # keyword 通道：先 BM25 打分，再查 doc_keywords 取 (token, field) 来源（spec [S5] 来源契约）。
    # 注：bm25_score 为文档级 BM25 分（同一文档的各 token 复读同一值，仅用于排序，非 token 级分）。
    from ..super_tree import KeywordIndex
    tokens = KeywordIndex._tokenize(None, query)
    if tokens:
        try:
            scored = dict(db.match_doc_keywords(tokens, top_k=topk))
        except Exception as e:
            logging.warning("evidence keyword scoring failed: %s", e)
            scored = {}
        if scored:
            tph = ",".join("?" for _ in tokens)
            dph = ",".join("?" for _ in scored)
            try:
                rows = db._connect().execute(
                    f"SELECT doc_id, keyword, field FROM doc_keywords "
                    f"WHERE keyword IN ({tph}) AND doc_id IN ({dph})",
                    (*tokens, *scored.keys())).fetchall()
                for r in rows:
                    entry(r["doc_id"])["channels"]["keyword"].append(
                        {"token": r["keyword"], "field": r["field"],
                         "bm25_score": scored.get(r["doc_id"], 0.0)})
            except Exception as e:
                logging.warning("evidence keyword provenance failed: %s", e)

    # tag 通道（closet 语义标签，source=llm）：closet_index 命中 doc 后，回填真实标签文本
    # （仅保留与 query token 子串匹配的标签，复用 UnifiedNodeEnhancement._tag_hits 语义）。
    if getattr(client, "closet_index", None):
        try:
            matched_docs = list(client.closet_index.search(query, top_k=topk))
        except Exception as e:
            logging.warning("evidence tag channel failed: %s", e)
            matched_docs = []
        for doc_id, score in matched_docs:
            texts = []
            try:
                tags = db.get_doc_tags(doc_id, source="llm")
                from .enhance import UnifiedNodeEnhancement
                hits = UnifiedNodeEnhancement._tag_hits(
                    tokens, query.casefold(), [t["tag_text"] for t in tags])
                hit_set = {h.casefold() for h in hits}
                texts = [(t["tag_text"], t["confidence"])
                         for t in tags if t["tag_text"].casefold() in hit_set]
            except Exception as e:
                logging.warning("evidence tag label lookup failed for doc %s: %s", doc_id, e)
            if texts:
                for tag_text, tag_conf in texts:
                    entry(doc_id)["channels"]["tag"].append(
                        {"text": tag_text, "confidence": float(tag_conf)})
            else:
                entry(doc_id)["channels"]["tag"].append(
                    {"text": "", "confidence": float(score)})

    # vector 通道：仅真向量后端（hybrid/chroma，is_vector=True）；keyword no-op 后端不进（防重复计分）
    backend = getattr(client, "search_backend", None)
    if backend is not None and getattr(backend, "is_vector", False):
        try:
            for doc_id, score in backend.search(query, top_k=topk):
                entry(doc_id)["channels"]["vector"].append({"score": float(score)})
        except Exception as e:
            logging.warning("evidence vector channel failed: %s", e)

    # entity 通道（query 实体 → 提及它们的文档）+ 图谱关联（CTE）
    try:
        entities = db.search_entities(query, limit=topk)
    except Exception as e:
        logging.warning("evidence entity search failed: %s", e)
        entities = []
    query_ids = [e["id"] for e in entities if e.get("id")]
    try:
        dist_table = db.get_entity_distances_cte(query_ids, max_hop=3) if query_ids else {}
    except Exception as e:
        logging.warning("evidence entity distance CTE failed: %s", e)
        dist_table = {}
    try:
        if query_ids:
            placeholders = ",".join("?" for _ in query_ids)
            rows = db._connect().execute(
                f"SELECT em.entity_id, em.doc_id, em.confidence, e.name, e.entity_type "
                f"FROM entity_mentions em JOIN entities e ON e.id = em.entity_id "
                f"WHERE em.entity_id IN ({placeholders})", query_ids).fetchall()
            for r in rows:
                entry(r["doc_id"])["channels"]["entity"].append(
                    {"name": r["name"], "type": r["entity_type"], "confidence": r["confidence"]})
    except Exception as e:
        logging.warning("evidence entity channel failed: %s", e)

    # graph 通道：CTE 邻居实体 → 提及它们的文档（邻居 id 来自 dist_table 的 key）
    if dist_table:
        try:
            neighbor_ids = list(dist_table.keys())
            placeholders = ",".join("?" for _ in neighbor_ids)
            rows = db._connect().execute(
                f"SELECT em.entity_id, em.doc_id, e.name "
                f"FROM entity_mentions em JOIN entities e ON e.id = em.entity_id "
                f"WHERE em.entity_id IN ({placeholders})", neighbor_ids).fetchall()
            for r in rows:
                info = dist_table[r["entity_id"]]
                entry(r["doc_id"])["graph"]["doc_entity_links"].append(
                    {"entity": r["name"], "distance": info["distance"],
                     "relation_type": info["relation_type"], "weight": info["weight"]})
        except Exception as e:
            logging.warning("evidence graph channel failed: %s", e)

    for db_id, e in bundle.items():
        e["channels"]["keyword"] = _dedup(e["channels"]["keyword"], lambda k: (k["token"], k["field"]))
        e["channels"]["entity"] = _dedup(e["channels"]["entity"], lambda k: k["name"])
        e["channels"]["tag"] = _dedup(e["channels"]["tag"], lambda k: k["text"])
    return bundle


def render_doc_evidence(bundle, db_ids) -> str:
    """L1 证据块：结构化呈现（按通道分组）。返回文本；空命中文档返回无证据注记。"""
    lines = []
    for db_id in db_ids:
        e = bundle.get(int(db_id))
        if not e:
            lines.append(f"doc {db_id}: 无通道命中")
            continue
        ch = e["channels"]
        parts = []
        if ch["keyword"]:
            parts.append("关键词命中: " + ", ".join(f"{k['token']}({k['field']})" for k in ch["keyword"]))
        if ch["entity"]:
            parts.append("实体命中: " + ", ".join(f"{x['name']}（{x['type']}）" for x in ch["entity"]))
        if ch["tag"]:
            parts.append("标签命中: " + ", ".join(t["text"] for t in ch["tag"]))
        links = e["graph"].get("doc_entity_links") or []
        if links:
            parts.append("图谱关联: " + ", ".join(
                f"{l['entity']}(距离{l['distance']}·{l['relation_type']})" for l in links))
        lines.append(f"doc {db_id}: " + " | ".join(parts) if parts else f"doc {db_id}: 无通道命中")
    return "\n".join(lines)
