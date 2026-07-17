/* ============================================================================
   components/upload-progress-list.js — Per-file upload progress rows.
   Driven by the uploadProgress ref from useUpload(). One row per in-flight
   or recently-completed file. State machine: queued → uploading → indexing
   → done | failed | cancelled.
   ========================================================================== */
(function (global) {
  "use strict";

  const UploadProgressList = {
    name: "UploadProgressList",
    props: {
      progressMap: { type: Object, required: true },
    },
    emits: ["cancel"],
    setup(props, { emit }) {
      function percent(p) {
        if (!p.total) return 0;
        return Math.round(((p.loaded || 0) / p.total) * 100);
      }
      function progressStatus(p) {
        if (p.phase === "failed") return "exception";
        if (p.phase === "done") return "success";
        if (p.phase === "cancelled") return "warning";
        return "";
      }
      function isIndeterminate(p) {
        return p.phase === "indexing" || p.phase === "queued";
      }
      function showText(p) {
        return p.phase !== "indexing" && p.phase !== "queued";
      }
      function canCancel(p) {
        return p.phase === "uploading" || p.phase === "queued" || p.phase === "indexing";
      }
      return { percent, progressStatus, isIndeterminate, showText, canCancel };
    },
    template: `
      <div v-if="Object.keys(progressMap).length" class="upload-progress-list">
        <div
          v-for="(p, name) in progressMap"
          :key="name"
          class="upload-row"
          :class="p.phase">
          <div class="upload-name" :title="name">{{ name }}</div>
          <el-progress
            class="upload-bar"
            :percentage="percent(p)"
            :status="progressStatus(p)"
            :indeterminate="isIndeterminate(p)"
            :show-text="showText(p)" />
          <div class="upload-phase">{{ p.phase }}</div>
          <el-button
            v-if="canCancel(p)"
            size="small"
            class="upload-cancel"
            @click="$emit('cancel', name)">取消</el-button>
          <div v-if="p.error" class="upload-error" :title="p.error">{{ p.error }}</div>
        </div>
      </div>
    `,
  };

  Object.assign(global, { UploadProgressList });
})(typeof window !== "undefined" ? window : globalThis);
