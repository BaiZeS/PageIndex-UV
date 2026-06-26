/* ============================================================================
   lib/config.js — model config form + connectivity self-test.
   T13 (FR12): testResult is null = never run; {status:"loading"} = in flight;
   {ok, latency_ms, error?, detail?, model?} = terminal. The alert binds to
   :type=:title from this object.
   ========================================================================== */
(function (global) {
  "use strict";

  const { ref, reactive } = Vue;

  function useConfig() {
    // Singleton: config snapshot + form + testResult must be shared
    // across the ConfigPanel (and any future consumers).
    if (global.__useConfigSingleton) return global.__useConfigSingleton;

    const config = ref(null);  // GET /api/config snapshot (null until loaded)
    const cfgForm = reactive({
      model: "",
      retrieve_model: "",
      api_key: "",
      base_url: "",
      persist: true,
    });
    const cfgMsg = ref("");
    const testResult = ref(null);

    async function loadConfig() {
      try {
        const c = await global.api("/api/config");
        config.value = c;
        // Edit-in-place: pre-fill inputs from the active snapshot.
        cfgForm.model = c.model || "";
        cfgForm.retrieve_model = c.retrieve_model || "";
        cfgForm.api_key = "";
        cfgForm.base_url = c.base_url || "";
        cfgForm.persist = true;
      } catch (e) {
        global.ElementPlus.ElMessage.error(e.message);
      }
    }

    async function saveConfig() {
      cfgMsg.value = "";
      const body = { persist: !!cfgForm.persist };
      if (cfgForm.model) body.model = cfgForm.model;
      if (cfgForm.retrieve_model) body.retrieve_model = cfgForm.retrieve_model;
      if (cfgForm.api_key) body.api_key = cfgForm.api_key;
      if (cfgForm.base_url) body.base_url = cfgForm.base_url;
      try {
        await global.api("/api/config", { method: "POST", body });
        cfgMsg.value = "已应用" + (body.persist ? " 并持久化" : "（仅运行时）");
        global.ElementPlus.ElMessage.success(cfgMsg.value);
        await loadConfig();
      } catch (e) {
        global.ElementPlus.ElMessage.error(e.message);
      }
    }

    /**
     * POST /api/config/test with the current form values (only the keys the
     * user actually filled in — empty fields are stripped so the backend
     * falls back to the active runtime config per spec §5.3).
     */
    async function testConnectivity() {
      testResult.value = { status: "loading" };
      try {
        const body = {};
        if (cfgForm.model) body.model = cfgForm.model;
        if (cfgForm.api_key) body.api_key = cfgForm.api_key;
        if (cfgForm.base_url) body.base_url = cfgForm.base_url;
        testResult.value = await global.api("/api/config/test", {
          method: "POST",
          body,
        });
      } catch (e) {
        testResult.value = { ok: false, error: "request_failed", detail: e.message };
      }
    }

    return global.__useConfigSingleton = {
      config, cfgForm, cfgMsg, testResult,
      loadConfig, saveConfig, testConnectivity,
    };
  }

  Object.assign(global, { useConfig });
})(typeof window !== "undefined" ? window : globalThis);
