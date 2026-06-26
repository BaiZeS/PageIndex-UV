const { createApp, ref, computed, onMounted, h } = Vue;

const API_KEY_STORE = "pageindex_api_key";

// Backend returns `confidence` as a string enum: "high" | "medium" | "low" | "unknown".
// Map it to a ratio so the confidence meter fills proportionally; display the
// label itself (never a fabricated percentage from the raw string).
const CONFIDENCE_RATIO = { high: 1, medium: 0.66, low: 0.33, unknown: 0 };
const CONFIDENCE_LABEL = { high: "高", medium: "中", low: "低", unknown: "未知" };

function apiKey() {
  return localStorage.getItem(API_KEY_STORE) || "";
}

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  const key = apiKey();
  if (key) headers["X-API-Key"] = key;
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

const app = createApp({
  setup() {
    const tab = ref("docs");
    const keyInput = ref(apiKey());
    // docs
    const docs = ref([]);
    const uploadResults = ref([]);
    // Per-file upload progress + XHR handles (T12, FR9). uploadProgress is the
    // reactive map T13 binds to <el-progress>; uploadXhrs holds the live
    // XMLHttpRequest instance so cancelUpload(name) can call .abort() (frontend
    // only — backend keeps processing the already-uploaded bytes per
    // A-RESOLVED-10).
    const uploadProgress = ref({});   // map: filename -> { loaded, total, phase, error }
    const uploadXhrs = ref({});       // map: filename -> XMLHttpRequest
    // search
    const query = ref("");
    const answer = ref(null);
    const searching = ref(false);
    // config
    const config = ref(null);
    const cfgForm = ref({ model: "", retrieve_model: "", api_key: "", base_url: "", persist: true });
    const cfgMsg = ref("");
    // T13 (FR12) — result of POST /api/config/test. null = never run;
    // {status:"loading"} = request in flight; {ok, latency_ms, error?, detail?}
    // = terminal. The UI binds the alert :type=:title to this.
    const testResult = ref(null);

    // Confidence is a backend string enum. Drive the meter from a mapped ratio
    // (never multiply the raw string). Confidence label shows the enum / a
    // clean human form, not a fabricated percentage.
    const confidenceRatio = computed(() => {
      const c = answer.value && answer.value.confidence;
      const key = typeof c === "string" ? c.toLowerCase() : "unknown";
      return CONFIDENCE_RATIO[key] != null ? CONFIDENCE_RATIO[key] : 0;
    });
    const confidenceLabel = computed(() => {
      const c = answer.value && answer.value.confidence;
      const key = typeof c === "string" ? c.toLowerCase() : "unknown";
      return CONFIDENCE_LABEL[key] || (c || "未知");
    });

    // matched_docs entries are enriched server-side with doc_name (preferred
    // source). Fall back to a client-side join against the loaded docs list
    // (id + doc_name) — defense-in-depth for any path that doesn't enrich —
    // then a short id. Avoids raw UUIDs in the ranked list.
    const matchedRows = computed(() => {
      const src = answer.value && answer.value.matched_docs;
      if (!src || !src.length) return [];
      const byId = {};
      for (const d of docs.value) byId[String(d.id)] = d.doc_name || d.doc_description || "";
      return src.map((d) => {
        const id = d.doc_id;
        const name = d.doc_name || byId[String(id)] || "";
        const shortId = id != null ? String(id) : "";
        return {
          key: id != null ? id : Math.random(),
          name: name || (shortId.length > 12 ? shortId.slice(0, 8) + "…" + shortId.slice(-4) : (shortId || "未知文档")),
          score: d.score,
        };
      });
    });

    const saveKey = () => { localStorage.setItem(API_KEY_STORE, keyInput.value); ElementPlus.ElMessage.success("Key saved"); };

    const loadDocs = async () => { try { docs.value = (await api("/api/documents")).documents; } catch (e) { ElementPlus.ElMessage.error(e.message); } };
    const onUpload = (opt) => {
      // T12 (FR9): XHR-based upload for byte-level progress + per-file phase
      // state machine. The backend /upload contract is unchanged (DD1-A):
      // synchronous single-POST returning {succeeded, total, results}. We get
      // real bytes-progress from `upload.onprogress`, then transition the
      // phase to "indexing" (indeterminate) the moment the bytes are fully
      // sent, and resolve to "done"/"failed" once the JSON response parses.
      const f = opt.file;
      const fd = new FormData();
      fd.append("file", f);
      const xhr = new XMLHttpRequest();
      uploadXhrs.value[f.name] = xhr;
      // Initialise progress entry so the UI can show 0% immediately. The
      // spread + overwrite pattern preserves any future fields cleanly.
      uploadProgress.value[f.name] = {
        loaded: 0, total: f.size, phase: "queued", error: null,
      };

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          uploadProgress.value[f.name] = {
            ...uploadProgress.value[f.name],
            loaded: e.loaded, total: e.total, phase: "uploading",
          };
        }
      };
      // Bytes are fully sent. Backend now indexes synchronously; we can't see
      // per-page progress (per DD1-A), so we hold at 99% and show an
      // indeterminate spinner in the "indexing" phase.
      xhr.upload.onload = () => {
        uploadProgress.value[f.name] = {
          ...uploadProgress.value[f.name],
          loaded: uploadProgress.value[f.name].total,
          phase: "indexing",
        };
      };
      xhr.onload = () => {
        try {
          const j = JSON.parse(xhr.responseText);
          const ok = (j.succeeded || 0) > 0;
          uploadProgress.value[f.name] = {
            ...uploadProgress.value[f.name],
            phase: ok ? "done" : "failed",
            error: ok ? null : (j.error || "unknown"),
          };
          // Prominent failure notification when any file in this response
          // failed to index (e.g. revoked LLM key → HTTP 200 with
          // succeeded:0). Mirrors prior behaviour.
          const okCount = (j.succeeded || 0);
          const totalCount = (j.total != null ? j.total : (j.results || []).length);
          if (okCount < totalCount) {
            const failedCount = totalCount - okCount;
            let firstErr = (j.results || []).find(r => !r.success)?.error || "";
            if (firstErr.length > 120) firstErr = firstErr.slice(0, 120) + "…";
            const message = totalCount <= 1
              ? f.name + (firstErr ? "：" + firstErr : "")
              : failedCount + "/" + totalCount + " 个文档索引失败" + (firstErr ? "：" + firstErr : "");
            ElementPlus.ElNotification({
              title: "文档索引失败", type: "error", duration: 6000, position: "top-right", message,
            });
          }
          uploadResults.value.push({ name: f.name, ok, raw: j });
          if (ok) loadDocs();
          ok ? opt.onSuccess() : opt.onError();
        } catch (e) {
          uploadProgress.value[f.name] = {
            ...uploadProgress.value[f.name],
            phase: "failed", error: e.message,
          };
          ElementPlus.ElMessage.error(e.message);
          opt.onError();
        } finally {
          delete uploadXhrs.value[f.name];
        }
      };
      xhr.onerror = () => {
        uploadProgress.value[f.name] = {
          ...uploadProgress.value[f.name],
          phase: "failed", error: "network error",
        };
        ElementPlus.ElMessage.error("网络错误");
        delete uploadXhrs.value[f.name];
        opt.onError();
      };
      // Abort is frontend-only per A-RESOLVED-10: backend doesn't see the
      // abort and will keep indexing whatever bytes already arrived.
      xhr.onabort = () => {
        uploadProgress.value[f.name] = {
          ...uploadProgress.value[f.name],
          phase: "cancelled", error: null,
        };
        delete uploadXhrs.value[f.name];
      };

      const key = apiKey();
      const h = {};
      if (key) h["X-API-Key"] = key;
      xhr.open("POST", "/upload");
      Object.entries(h).forEach(([k, v]) => xhr.setRequestHeader(k, v));
      xhr.send(fd);
    };

    // Cancel an in-flight upload by filename. Frontend-only — backend keeps
    // processing. Safe to call when no XHR exists (no-op).
    const cancelUpload = (filename) => {
      const xhr = uploadXhrs.value[filename];
      if (xhr) xhr.abort();
    };

    // ── T13 (FR10) — deleteDoc(doc) ───────────────────────────────────────
    // Confirms via ElMessageBox then DELETE /api/documents/{id}, reloads the
    // list on success. Any thrown error surfaces as a toast (delete path
    // integrity is W2's territory; we just call the endpoint and trust the
    // cascade).
    const deleteDoc = async (doc) => {
      try {
        await ElementPlus.ElMessageBox.confirm(
          `删除文档 "${doc.doc_name || doc.pdf_name || doc.id}"?此操作不可恢复`,
          "确认删除",
          {
            confirmButtonText: "删除",
            cancelButtonText: "取消",
            type: "warning",
            confirmButtonClass: "el-button--danger",
          }
        );
      } catch (_) {
        return; // user cancelled — ElMessageBox rejects on cancel
      }
      try {
        await api(`/api/documents/${doc.id}`, { method: "DELETE" });
        ElementPlus.ElMessage.success("文档已删除");
        await loadDocs();
      } catch (e) {
        ElementPlus.ElMessage.error(`删除失败: ${e.message}`);
      }
    };

    // ── T13 (FR11) — evidenceModule(answer) ────────────────────────────────
    // Render a <el-collapse> per matched_doc, each item showing the doc's
    // selected_nodes (title + path + truncated summary + page numbers) and
    // any pages that belong to that doc. Top header shows the N/M/X summary
    // ("命中 N 篇 · M 节点 · X 页"). Defensive against missing fields
    // (R-DD3-1: backend may enrich only partially in v1.0 paths).
    const evidenceModule = (answer) => {
      const md = (answer && answer.matched_docs) || [];
      const nodes = (answer && answer.selected_nodes) || [];
      const pages = (answer && answer.pages) || [];
      const summary = `命中 ${md.length} 篇 · ${nodes.length} 节点 · ${pages.length} 页`;
      if (!md.length) {
        return h("div", { class: "evidence-module" }, [
          h("div", { class: "evidence-summary" }, summary),
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
          ElementPlus.ElCollapseItem,
          { key: k || i, name: String(k || i), title },
          () => [
            h("div", { class: "evidence-section-title" }, `节点 (${docNodes.length})`),
            h(
              "div",
              { class: "evidence-nodes" },
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
                        ? h(
                            "div",
                            { class: "evidence-node-summary" },
                            summary2.length > 200 ? summary2.slice(0, 200) + "…" : summary2
                          )
                        : null,
                      nodePages.length
                        ? h(
                            "div",
                            { class: "evidence-node-pages" },
                            "页码: " + (Array.isArray(nodePages) ? nodePages.join(", ") : nodePages)
                          )
                        : null,
                    ]);
                  })
                : [h("div", { class: "evidence-empty-sub" }, "—")]
            ),
            h("div", { class: "evidence-section-title" }, `页码片段 (${docPages.length})`),
            h(
              "div",
              { class: "evidence-pages" },
              docPages.length
                ? docPages.map((p, j) => {
                    const pageNo = p.page != null ? p.page : p.page_number != null ? p.page_number : j + 1;
                    const txt = p.text || p.snippet || "";
                    return h("div", { class: "evidence-page", key: j }, [
                      h("span", { class: "evidence-page-num" }, `第 ${pageNo} 页`),
                      txt
                        ? h(
                            "span",
                            { class: "evidence-page-text" },
                            txt.length > 200 ? txt.slice(0, 200) + "…" : txt
                          )
                        : null,
                    ]);
                  })
                : [h("div", { class: "evidence-empty-sub" }, "—")]
            ),
          ]
        );
      });
      return h("div", { class: "evidence-module" }, [
        h("div", { class: "evidence-summary" }, summary),
        h(
          ElementPlus.ElCollapse,
          { modelValue: [], accordion: true },
          () => items
        ),
      ]);
    };

    // ── T13 (FR12) — testConnectivity() ────────────────────────────────────
    // POST /api/config/test with the current form values (only the keys the
    // user actually filled in — empty fields are stripped so the backend
    // falls back to the active runtime config per spec §5.3). The button is
    // loading-bound to testResult.value.status === "loading".
    const testConnectivity = async () => {
      testResult.value = { status: "loading" };
      try {
        const body = {};
        if (cfgForm.value.model) body.model = cfgForm.value.model;
        if (cfgForm.value.api_key) body.api_key = cfgForm.value.api_key;
        if (cfgForm.value.base_url) body.base_url = cfgForm.value.base_url;
        const r = await api("/api/config/test", {
          method: "POST",
          body: JSON.stringify(body),
        });
        testResult.value = r;
      } catch (e) {
        testResult.value = { ok: false, error: "request_failed", detail: e.message };
      }
    };

    const doSearch = async () => {
      if (!query.value.trim()) return; searching.value = true; answer.value = null;
      try { answer.value = await api("/api/search", { method: "POST", body: JSON.stringify({ query: query.value, top_k: 5 }) }); }
      catch (e) { ElementPlus.ElMessage.error(e.message); } finally { searching.value = false; }
    };
    // Config form is edit-in-place: inputs are pre-filled with the current
    // values from GET /api/config (empty until loaded).
    const loadConfig = async () => { try { config.value = await api("/api/config"); const c = config.value; cfgForm.value.model = c.model||""; cfgForm.value.retrieve_model = (c.retrieve_model||""); } catch (e) { ElementPlus.ElMessage.error(e.message); } };
    const saveConfig = async () => {
      cfgMsg.value = "";
      const body = { persist: cfgForm.value.persist };
      if (cfgForm.value.model) body.model = cfgForm.value.model;
      if (cfgForm.value.retrieve_model) body.retrieve_model = cfgForm.value.retrieve_model;
      if (cfgForm.value.api_key) body.api_key = cfgForm.value.api_key;
      if (cfgForm.value.base_url) body.base_url = cfgForm.value.base_url;
      try { await api("/api/config", { method: "POST", body: JSON.stringify(body) }); cfgMsg.value = "已应用" + (body.persist ? "并持久化" : "（仅运行时）"); ElementPlus.ElMessage.success(cfgMsg.value); loadConfig(); }
      catch (e) { ElementPlus.ElMessage.error(e.message); }
    };

    onMounted(() => { loadDocs(); loadConfig(); });
    return {
      tab, keyInput, saveKey, docs, uploadResults, onUpload,
      uploadProgress, uploadXhrs, cancelUpload, deleteDoc,
      query, answer, searching, doSearch, evidenceModule,
      confidenceRatio, confidenceLabel, matchedRows,
      config, cfgForm, cfgMsg, testResult, testConnectivity, saveConfig, loadConfig, loadDocs,
    };
  },
});
app.use(ElementPlus);
app.mount("#app");
