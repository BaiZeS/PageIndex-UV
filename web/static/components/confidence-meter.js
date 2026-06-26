/* ============================================================================
   components/confidence-meter.js — Confidence bar + label chip.
   Pure presentational. Source of truth is the answer.confidence string
   enum (high/medium/low/unknown); never fabricate a percentage from it.
   ========================================================================== */
(function (global) {
  "use strict";

  const ConfidenceMeter = {
    name: "ConfidenceMeter",
    props: {
      confidence: { type: String, default: "unknown" },
    },
    setup(props) {
      return {
        ratio:    () => Math.round(global.confidenceRatio(props.confidence) * 100),
        label:    () => global.confidenceLabel(props.confidence),
        cssClass: () => global.confidenceClass(props.confidence),
      };
    },
    template: `
      <div class="answer-meta">
        <span class="conf-label">置信度 · Confidence</span>
        <div class="conf-meter" :title="label">
          <div class="conf-bar" :style="{ width: ratio() + '%' }"></div>
        </div>
        <span class="conf-val" :class="cssClass()">{{ label }}</span>
      </div>
    `,
  };

  Object.assign(global, { ConfidenceMeter });
})(typeof window !== "undefined" ? window : globalThis);
