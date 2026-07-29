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
   * Format score as percentage with color class
   */
  function formatScore(score) {
    if (score == null) return { text: "-", cls: "low" };
    const pct = Math.round(score * 100);
    if (pct >= 80) return { text: `${pct}%`, cls: "high" };
    if (pct >= 50) return { text: `${pct}%`, cls: "medium" };
    return { text: `${pct}%`, cls: "low" };
  }

  /**
   * Truncate text with expand/collapse support
   */
  function truncateText(text, maxLen = 200) {
    if (!text || text.length <= maxLen) return text;
    return text.slice(0, maxLen);
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

    if (!md.length) {
      return h("div", { class: "evidence-module" }, [
        h("div", { class: "evidence-empty" }, [
          h("div", { class: "evidence-empty-icon" }, "🔍"),
          h("div", { class: "evidence-empty-text" }, "未找到相关证据"),
        ]),
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
      const scoreInfo = formatScore(d.score);
      const docTitle = d.doc_name || d.doc_id || `文档 ${i + 1}`;

      return h(
        global.ElementPlus.ElCollapseItem,
        { key: k || i, name: String(k || i) },
        {
          title: () => h("div", { class: "evidence-doc-header" }, [
            h("div", { class: "evidence-doc-title" }, [
              h("span", { class: "evidence-doc-icon" }, "📄"),
              h("span", { class: "evidence-doc-name" }, docTitle),
            ]),
            h("div", { class: "evidence-doc-meta" }, [
              h("span", { class: `evidence-score ${scoreInfo.cls}` }, scoreInfo.text),
              h("span", { class: "evidence-doc-count" }, `${docNodes.length} 节点`),
              h("span", { class: "evidence-doc-divider" }, "·"),
              h("span", { class: "evidence-doc-count" }, `${docPages.length} 页`),
            ]),
          ]),
          default: () => [
            // Nodes section
            docNodes.length > 0 ? h("div", { class: "evidence-section" }, [
              h("div", { class: "evidence-section-header" }, [
                h("span", { class: "evidence-section-icon" }, "📑"),
                h("span", { class: "evidence-section-title" }, "匹配节点"),
                h("span", { class: "evidence-section-count" }, docNodes.length),
              ]),
              h("div", { class: "evidence-nodes" },
                docNodes.map((n, j) => {
                  const t = n.title || n.node_id || `节点 ${j + 1}`;
                  const path = n.path || n.node_path || "";
                  const summaryText = n.summary || n.text || n.snippet || "";
                  const nodePages = n.pages || n.page_numbers || [];
                  const isLong = summaryText.length > 200;
                  return h("div", { class: "evidence-node", key: j }, [
                    h("div", { class: "evidence-node-header" }, [
                      h("div", { class: "evidence-node-title" }, [
                        h("span", { class: "evidence-node-icon" }, "▎"),
                        h("span", null, t),
                      ]),
                      nodePages.length
                        ? h("div", { class: "evidence-node-pages" },
                            nodePages.map((p, idx) => h("span", { class: "evidence-page-tag", key: idx }, `P${p}`)))
                        : null,
                    ]),
                    path ? h("div", { class: "evidence-node-path" }, path) : null,
                    summaryText
                      ? h("div", { class: "evidence-node-summary" }, [
                          h("span", null, isLong ? truncateText(summaryText) + "…" : summaryText),
                          isLong ? h("span", { class: "evidence-expand-hint" }, "更多") : null,
                        ])
                      : null,
                  ]);
                })
              ),
            ]) : null,

            // Pages section
            docPages.length > 0 ? h("div", { class: "evidence-section" }, [
              h("div", { class: "evidence-section-header" }, [
                h("span", { class: "evidence-section-icon" }, "📃"),
                h("span", { class: "evidence-section-title" }, "相关页面"),
                h("span", { class: "evidence-section-count" }, docPages.length),
              ]),
              h("div", { class: "evidence-pages" },
                docPages.map((p, j) => {
                  const pageNo = p.page != null ? p.page
                    : p.page_number != null ? p.page_number : j + 1;
                  const txt = p.text || p.snippet || "";
                  const isLong = txt.length > 200;
                  return h("div", { class: "evidence-page", key: j }, [
                    h("div", { class: "evidence-page-badge" }, [
                      h("span", { class: "evidence-page-num" }, `${pageNo}`),
                      h("span", { class: "evidence-page-label" }, "页"),
                    ]),
                    h("div", { class: "evidence-page-content" }, [
                      h("span", null, isLong ? truncateText(txt) + "…" : txt),
                      isLong ? h("span", { class: "evidence-expand-hint" }, "更多") : null,
                    ]),
                  ]);
                })
              ),
            ]) : null,

            // Empty state for this doc
            docNodes.length === 0 && docPages.length === 0
              ? h("div", { class: "evidence-empty-sub" }, "暂无详细内容")
              : null,
          ],
        }
      );
    });

    return h("div", { class: "evidence-module" }, [
      // Summary bar
      h("div", { class: "evidence-summary" }, [
        h("div", { class: "evidence-summary-left" }, [
          h("span", { class: "evidence-summary-icon" }, "🔗"),
          h("span", { class: "evidence-summary-label" }, "溯源证据"),
        ]),
        h("div", { class: "evidence-summary-stats" }, [
          h("div", { class: "evidence-stat" }, [
            h("span", { class: "evidence-stat-value" }, md.length),
            h("span", { class: "evidence-stat-label" }, "文档"),
          ]),
          h("div", { class: "evidence-stat-divider" }),
          h("div", { class: "evidence-stat" }, [
            h("span", { class: "evidence-stat-value" }, nodes.length),
            h("span", { class: "evidence-stat-label" }, "节点"),
          ]),
          h("div", { class: "evidence-stat-divider" }),
          h("div", { class: "evidence-stat" }, [
            h("span", { class: "evidence-stat-value" }, pages.length),
            h("span", { class: "evidence-stat-label" }, "页面"),
          ]),
        ]),
      ]),
      // Document list
      h(global.ElementPlus.ElCollapse,
        { modelValue: [], accordion: true, class: "evidence-collapse" },
        () => items),
    ]);
  }

  Object.assign(global, { useQA, evidenceModule });
})(typeof window !== "undefined" ? window : globalThis);
