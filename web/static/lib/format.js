/* ============================================================================
   lib/format.js — small pure formatters shared across components.
   ========================================================================== */
(function (global) {
  "use strict";

  // Confidence is a backend string enum. Map to a ratio (never fabricate
  // a percentage from the raw string) and a display label.
  const CONFIDENCE_RATIO = { high: 1, medium: 0.66, low: 0.33, unknown: 0 };
  const CONFIDENCE_LABEL = { high: "高", medium: "中", low: "低", unknown: "未知" };

  function confidenceRatio(value) {
    const k = typeof value === "string" ? value.toLowerCase() : "unknown";
    return CONFIDENCE_RATIO[k] != null ? CONFIDENCE_RATIO[k] : 0;
  }

  function confidenceLabel(value) {
    const k = typeof value === "string" ? value.toLowerCase() : "unknown";
    return CONFIDENCE_LABEL[k] || value || "未知";
  }

  function confidenceClass(value) {
    const k = typeof value === "string" ? value.toLowerCase() : "unknown";
    return k; // "high" | "medium" | "low" | "unknown" — matches .conf-val.<k>
  }

  // Truncate a string to n chars with a trailing ellipsis when needed.
  function truncate(s, n) {
    if (s == null) return "";
    const str = String(s);
    return str.length > n ? str.slice(0, n) + "…" : str;
  }

  // Format bytes for human-readable display (1024-based).
  function formatBytes(bytes) {
    if (bytes == null || isNaN(bytes)) return "—";
    const units = ["B", "KB", "MB", "GB"];
    let n = Number(bytes);
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i++;
    }
    return (i === 0 ? n.toFixed(0) : n.toFixed(n < 10 ? 1 : 0)) + " " + units[i];
  }

  Object.assign(global, {
    confidenceRatio, confidenceLabel, confidenceClass,
    truncate, formatBytes,
  });
})(typeof window !== "undefined" ? window : globalThis);
