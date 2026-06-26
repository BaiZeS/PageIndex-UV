/* ============================================================================
   components/panel.js — Reusable panel shell with eyebrow + title + slot
   for action buttons + default slot for body.
   ========================================================================== */
(function (global) {
  "use strict";

  const Panel = {
    name: "Panel",
    props: {
      title: { type: String, required: true },
      eyebrow: { type: String, default: "" },
      desc: { type: String, default: "" },
      narrow: { type: Boolean, default: false },
    },
    template: `
      <section class="panel" :class="{ 'panel--narrow': narrow }">
        <header class="panel-head">
          <div>
            <div v-if="eyebrow" class="section-eyebrow">{{ eyebrow }}</div>
            <h2 class="section-title">{{ title }}</h2>
            <p v-if="desc" class="section-desc">{{ desc }}</p>
          </div>
          <slot name="actions" />
        </header>
        <slot />
      </section>
    `,
  };

  Object.assign(global, { Panel });
})(typeof window !== "undefined" ? window : globalThis);
