const { createApp, ref, computed, onMounted } = Vue;

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
    // search
    const query = ref("");
    const answer = ref(null);
    const searching = ref(false);
    // config
    const config = ref(null);
    const cfgForm = ref({ model: "", retrieve_model: "", api_key: "", base_url: "", persist: true });
    const cfgMsg = ref("");

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
    const onUpload = async (opt) => {
      const f = opt.file; const fd = new FormData(); fd.append("file", f);
      try {
        const key = apiKey(); const h = {}; if (key) h["X-API-Key"] = key;
        const r = await fetch("/upload", { method: "POST", headers: h, body: fd });
        // Mirror api()'s non-OK check so a 403/413 surfaces the server's error
        // instead of being swallowed as a silent {ok:false} row.
        if (!r.ok) {
          let msg = `HTTP ${r.status}`;
          try { msg = (await r.json()).error || msg; } catch (e) {}
          throw new Error(msg);
        }
        const j = await r.json();
        uploadResults.value.push({ name: f.name, ok: (j.succeeded||0)>0, raw: j });
        if ((j.succeeded||0) > 0) loadDocs();
        opt.onSuccess();
      } catch (e) { ElementPlus.ElMessage.error(e.message); opt.onError(); }
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
      query, answer, searching, doSearch,
      confidenceRatio, confidenceLabel, matchedRows,
      config, cfgForm, cfgMsg, saveConfig, loadConfig, loadDocs,
    };
  },
});
app.use(ElementPlus);
app.mount("#app");
