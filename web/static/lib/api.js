/* ============================================================================
   lib/api.js — fetch wrapper + auth header
   Pure helpers, no Vue. Exposes api(), apiKey(), setApiKey(), hasApiKey().
   ========================================================================== */
(function (global) {
  "use strict";

  const API_KEY_STORE = "pageindex_api_key";

  function apiKey() {
    return localStorage.getItem(API_KEY_STORE) || "";
  }

  function setApiKey(v) {
    if (v) localStorage.setItem(API_KEY_STORE, v);
    else localStorage.removeItem(API_KEY_STORE);
  }

  function hasApiKey() {
    return Boolean(apiKey());
  }

  /**
   * Fetch wrapper that auto-attaches the X-API-Key header and unwraps
   * {error} responses into thrown Error messages so callers can
   * try/catch uniformly. Body is auto-JSON-encoded when given as an object.
   */
  async function api(path, opts = {}) {
    const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    const key = apiKey();
    if (key) headers["X-API-Key"] = key;

    const init = { ...opts, headers };
    if (init.body && typeof init.body !== "string") {
      init.body = JSON.stringify(init.body);
    }

    const res = await fetch(path, init);
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j && j.error) msg = j.error;
      } catch (e) {
        /* response had no JSON body — keep status-line message */
      }
      throw new Error(msg);
    }
    return res.json();
  }

  Object.assign(global, { api, apiKey, setApiKey, hasApiKey });
})(typeof window !== "undefined" ? window : globalThis);
