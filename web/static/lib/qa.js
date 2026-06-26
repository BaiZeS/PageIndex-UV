/* ============================================================================
   lib/qa.js — Q&A state + evidence chain rendering.
   Two pieces: useQA() for search state, and evidenceModule() for the
   collapsible per-doc evidence view.
   ========================================================================== */
(function (global) {
  "use strict";

  const { ref, computed, h } = Vue;

  function useQA(docsRef) {
    // Singleton: same singleton discipline as useUpload — keep query /
    // answer / searching shared across components.
    if (global.__useQASingleton) return global.__useQASingleton;

    const query = ref("");
    const answer = ref(null);
    const searching = ref(false);

    async function doSearch() {
      const q = (query.value || "").trim();
      if (!q) return;
      searching.value = true;
      answer.value = null;
      try {
        answer.value = await global.api("/api/search", {
          method: "POST",
          body: { query: q, top_k: 5 },
        });
      } catch (e) {
        global.ElementPlus.ElMessage.error(e.message);
      } finally {
        searching.value = false;
      }
    }

    return global.__useQASingleton = { query, answer, searching, doSearch };
  }

  /**
   * Render a <el-collapse> per matched_doc, each item showing the doc's
   * selected_nodes (title + path + truncated summary + page numbers) and
   * any pages that belong to that doc. Top header shows the N/M/X summary
   * ("命中 N 篇 · M 节点 · X 页"). Defensive against missing fields
   * (R-DD3-1: backend may enrich only partially in v1.0 paths).
   *
   * Kept as a function (not a component) so it can be plugged in via
   * <component :is="evidenceModule(answer)" v-if="answer" /> — same
   * mounting mechanism the original used.
   */
  function evidenceModule(answer) {
    const md = (answer && answer.matched_docs) || [];
    const nodes = (answer && answer.selected_nodes) || [];
    const pages = (answer && answer.pages) || [];
    const summary = `命中 ${md.length} 篇 · ${nodes.length} 节点 · ${pages.length} 页`;

    if (!md.length) {
      return h("div", { class: "evidence-module" }, [
        h("div", { class: "evidence-summary" }, [
          h("span", null, "Evidence"),
          h("span", { class: "dot" }, "·"),
          h("span", null, summary),
        ]),
        h("div", { class: "evidence-empty" }, "未找到相关证据"),
      ]);
    }

    // Group nodes/pages by doc_id (or pdf_name as a soft key for legacy rows
    // that don't carry doc_id). Pages without a doc_id get attached to the
    // first matched doc so they're never silently dropped from the UI.
    const groupByDoc = (items) => {
      const byKey = {};
      for (const it of items || []) {
        const k = it.doc_id != null ? String(it.doc_id) : (it.pdf_name || "");
        if (!byKey[k]) byKey[k] = [];
        byKey[k].push(it);
      }
      return byKey;
    };

    const nodesByDoc = groupByDoc(nodes);
    const pagesByDoc = groupByDoc(pages);

    const items = md.map((d, i) => {
      const k = d.doc_id != null ? String(d.doc_id) : (d.doc_name || d.pdf_name || "");
      const docNodes = nodesByDoc[k] || nodesByDoc[d.doc_name] || [];
      const docPages = pagesByDoc[k] || pagesByDoc[d.doc_name] || [];
      const title = `${d.doc_name || d.doc_id || "doc-" + i} · 匹配度 ${d.score != null ? d.score : "-"}`;
      return h(
        global.ElementPlus.ElCollapseItem,
        { key: k || i, name: String(k || i), title },
        () => [
          h("div", { class: "evidence-section-title" }, [
            h("span", null, "节点"),
            h("span", { class: "count" }, `(${docNodes.length})`),
          ]),
          h("div", { class: "evidence-nodes" },
            docNodes.length
              ? docNodes.map((n, j) => {
                  const t = n.title || n.node_id || `节点 ${j + 1}`;
                  const path = n.path || n.node_path || "";
                  const summary2 = n.summary || n.text || n.snippet || "";
                  const nodePages = n.pages || n.page_numbers || [];
                  return h("div", { class: "evidence-node", key: j }, [
                    h("div", { class: "evidence-node-title" }, t),
                    path ? h("div", { class: "evidence-node-path" }, path) : null,
                    summary2
                      ? h("div", { class: "evidence-node-summary" },
                          summary2.length > 200 ? summary2.slice(0, 200) + "…" : summary2)
                      : null,
                    nodePages.length
                      ? h("div", { class: "evidence-node-pages" },
                          "页码: " + (Array.isArray(nodePages) ? nodePages.join(", ") : nodePages))
                      : null,
                  ]);
                })
              : [h("div", { class: "evidence-empty-sub" }, "—")]
          ),
          h("div", { class: "evidence-section-title" }, [
            h("span", null, "页码片段"),
            h("span", { class: "count" }, `(${docPages.length})`),
          ]),
          h("div", { class: "evidence-pages" },
            docPages.length
              ? docPages.map((p, j) => {
                  const pageNo = p.page != null ? p.page
                    : p.page_number != null ? p.page_number : j + 1;
                  const txt = p.text || p.snippet || "";
                  return h("div", { class: "evidence-page", key: j }, [
                    h("span", { class: "evidence-page-num" }, `第 ${pageNo} 页`),
                    txt
                      ? h("span", { class: "evidence-page-text" },
                          txt.length > 200 ? txt.slice(0, 200) + "…" : txt)
                      : null,
                  ]);
                })
              : [h("div", { class: "evidence-empty-sub" }, "—")]
          ),
        ]
      );
    });

    return h("div", { class: "evidence-module" }, [
      h("div", { class: "evidence-summary" }, [
        h("span", null, "Evidence"),
        h("span", { class: "dot" }, "·"),
        h("span", { class: "count" }, md.length),
        h("span", null, "篇"),
        h("span", { class: "dot" }, "·"),
        h("span", { class: "count" }, nodes.length),
        h("span", null, "节点"),
        h("span", { class: "dot" }, "·"),
        h("span", { class: "count" }, pages.length),
        h("span", null, "页"),
      ]),
      h(global.ElementPlus.ElCollapse,
        { modelValue: [], accordion: true, class: "evidence-collapse" },
        () => items),
    ]);
  }

  Object.assign(global, { useQA, evidenceModule });
})(typeof window !== "undefined" ? window : globalThis);
