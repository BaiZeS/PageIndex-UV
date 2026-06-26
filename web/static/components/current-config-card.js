/* ============================================================================
   components/current-config-card.js — "Current" snapshot card for config tab.
   Shows model · provider · base_url · key status as inline mono fields.
   ========================================================================== */
(function (global) {
  "use strict";

  const CurrentConfigCard = {
    name: "CurrentConfigCard",
    props: {
      config: { type: Object, default: null },
    },
    template: `
      <div v-if="config" class="current-card">
        <span class="current-label">当前</span>
        <span class="current-field"><em>model</em> {{ config.model || '—' }}</span>
        <span class="dot">·</span>
        <span class="current-field">{{ config.provider || '—' }}</span>
        <span class="dot">·</span>
        <span class="current-field mono">{{ config.base_url || '—' }}</span>
        <span class="dot">·</span>
        <span class="current-field" :class="{ present: config.has_api_key }">
          key {{ config.has_api_key ? ('已设置 ' + (config.api_key_masked || '')) : '未设置' }}
        </span>
      </div>
    `,
  };

  Object.assign(global, { CurrentConfigCard });
})(typeof window !== "undefined" ? window : globalThis);
