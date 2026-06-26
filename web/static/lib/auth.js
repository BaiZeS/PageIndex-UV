/* ============================================================================
   lib/auth.js — API-key input state + persistence.
   Wraps the localStorage helpers from api.js with a Vue ref.
   ========================================================================== */
(function (global) {
  "use strict";

  const { ref, watch } = Vue;

  function useAuth() {
    if (global.__useAuthSingleton) return global.__useAuthSingleton;

    const keyInput = ref(global.apiKey());
    const hasKey = ref(global.hasApiKey());

    function saveKey() {
      const v = (keyInput.value || "").trim();
      global.setApiKey(v);
      hasKey.value = Boolean(v);
      global.ElementPlus.ElMessage.success("Key 已保存");
      // After saving a valid key, refetch data so the UI updates without
      // a manual page reload. Guards against singleton-not-yet-created and
      // against the request failing (e.g. wrong key) — failures surface
      // through the fetch's own error path, not this caller.
      if (v) {
        try {
          global.useUpload().loadDocs();
          global.useConfig().loadConfig();
        } catch (_) {
          /* singleton may not be created yet — ignore */
        }
      }
    }

    // Keep the input in sync if anything else clears/sets the key
    // (e.g. future "logout" affordance). One-way: storage → input.
    watch(hasKey, (v) => {
      keyInput.value = v ? global.apiKey() : "";
    });

    return global.__useAuthSingleton = { keyInput, hasKey, saveKey };
  }

  Object.assign(global, { useAuth });
})(typeof window !== "undefined" ? window : globalThis);
