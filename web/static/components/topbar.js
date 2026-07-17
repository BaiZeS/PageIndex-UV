/* ============================================================================
   components/topbar.js — Brand mark + API key input.
   Pure presentational. State lives in useAuth().
   ========================================================================== */
(function (global) {
  "use strict";

  const Topbar = {
    name: "Topbar",
    setup() {
      const { keyInput, hasKey, saveKey } = global.useAuth();
      return { keyInput, hasKey, saveKey };
    },
    template: `
      <header class="app-topbar">
        <div class="topbar-inner">
          <div class="brand">
            <span class="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 20 20" width="18" height="18" fill="none">
                <rect x="1"  y="1"  width="5" height="5" rx="1" fill="currentColor"/>
                <rect x="7.5" y="1"  width="5" height="5" rx="1" fill="currentColor" opacity="0.32"/>
                <rect x="14" y="1"  width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.2"/>
                <rect x="1"  y="7.5" width="5" height="5" rx="1" fill="currentColor" opacity="0.32"/>
                <rect x="7.5" y="7.5" width="5" height="5" rx="1" fill="currentColor"/>
                <rect x="14" y="7.5" width="5" height="5" rx="1" fill="currentColor" opacity="0.32"/>
                <rect x="1"  y="14" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.2"/>
                <rect x="7.5" y="14" width="5" height="5" rx="1" fill="currentColor" opacity="0.32"/>
                <rect x="14" y="14" width="5" height="5" rx="1" fill="currentColor"/>
              </svg>
            </span>
            <span class="brand-name">PageIndex</span>
            <span class="brand-sub">Console · 控制台</span>
          </div>
          <div class="auth">
            <span
              class="auth-status"
              :class="{ empty: !hasKey }"
              :title="hasKey ? 'API Key 已设置' : 'API Key 未设置'"
              aria-hidden="true"></span>
            <el-input
              v-model="keyInput"
              placeholder="API Key（与 .env API_KEY 一致）"
              class="auth-input"
              show-password
              size="default"
              @keyup.enter="saveKey">
              <template #prefix><span class="auth-key-icon">key</span></template>
            </el-input>
            <el-button type="primary" class="auth-save" @click="saveKey">保存</el-button>
          </div>
        </div>
      </header>
    `,
  };

  Object.assign(global, { Topbar });
})(typeof window !== "undefined" ? window : globalThis);
