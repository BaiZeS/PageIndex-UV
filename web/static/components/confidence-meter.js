/* ============================================================================
   components/confidence-meter.js — Confidence bar + label chip.
   Pure presentational. Source of truth is the answer.confidence string
   enum (high/medium/low/unknown); never fabricate a percentage from it.
   ========================================================================== */
(function (global) {
  "use strict";

  const { computed } = Vue;

  const ConfidenceMeter = {
    name: "ConfidenceMeter",
    props: {
      confidence: { type: String, default: "unknown" },
    },
    setup(props) {
      // Computed refs (not bare functions) — gives us caching + correct
      // reactive dependency tracking on props.confidence. Bare functions
      // returned from setup would re-execute on every template access.
      const ratio    = computed(() => Math.round(global.confidenceRatio(props.confidence) * 100));
      const label    = computed(() => global.confidenceLabel(props.confidence));
      const cssClass = computed(() => global.confidenceClass(props.confidence));
      return { ratio, label, cssClass };
    },
    template: `
      <div class="answer-meta">
        <span class="conf-label">置信度 · Confidence</span>
        <div class="conf-meter" :title="label">
          <div class="conf-bar" :style="{ width: ratio + '%' }"></div>
        </div>
        <span class="conf-val" :class="cssClass">{{ label }}</span>
      </div>
    `,
  };

  Object.assign(global, { ConfidenceMeter });
})(typeof window !== "undefined" ? window : globalThis);
