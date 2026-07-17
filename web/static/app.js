/* ============================================================================
   app.js — Bootstrap only.
   All business logic lives in /static/lib/. All presentation lives in
   /static/components/. This file's only job is to wire the pieces together.
   ========================================================================== */
(function (global) {
  "use strict";

  const { createApp, ref } = Vue;

  const app = createApp({
    // Root template lives in JS (not in index.html) so Vue uses its full
    // compiler — which supports self-closing custom components like
    // <topbar />. If we left this in index.html, the HTML parser would
    // re-parent <main> as a child of <topbar>, and the Topbar component
    // (which has no <slot />) would drop it. The bug-fix report
    // docs/devkit/cases/2026-06-26-web-empty-page.md has the full
    // mechanical proof from @vue/compiler-core source.
    template: `
      <topbar />
      <main class="app-main">
        <tab-nav v-model="tab">
          <template #docs><docs-panel /></template>
          <template #qa><qa-panel /></template>
          <template #cfg><config-panel /></template>
        </tab-nav>
      </main>
    `,
    setup() {
      // The active tab is the only piece of UI state that lives above
      // individual components. Everything else is owned by composables.
      const tab = ref("docs");
      return { tab };
    },
  });

  // Register components globally so <topbar />, <tab-nav> etc. work in the
  // template without per-component imports.
  app.component("topbar", global.Topbar);
  app.component("tab-nav", global.TabNav);
  app.component("panel", global.Panel);
  app.component("docs-panel", global.DocsPanel);
  app.component("qa-panel", global.QAPanel);
  app.component("config-panel", global.ConfigPanel);
  app.component("confidence-meter", global.ConfidenceMeter);
  app.component("upload-progress-list", global.UploadProgressList);
  app.component("current-config-card", global.CurrentConfigCard);

  app.use(ElementPlus);

  // Kick off initial loads after mount so the panels can read populated
  // state. Both loadDocs and loadConfig are idempotent and tolerate
  // concurrent invocation.
  app.mount("#app");
  // Only fetch data on mount if an API key is already saved. On a fresh
  // visit (no localStorage key) we deliberately skip — fetching now would
  // 403 and surface two red "Invalid or missing API Key" toasts before the
  // user has a chance to type one. After the user saves a key in the
  // topbar, useAuth().saveKey() triggers the same loadDocs/loadConfig
  // calls. The key check is a global lookup (cheap) and the fetches
  // themselves remain idempotent.
  if (global.apiKey()) {
    global.useUpload().loadDocs();
    global.useConfig().loadConfig();
  }
})(typeof window !== "undefined" ? window : globalThis);
