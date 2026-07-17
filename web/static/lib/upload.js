/* ============================================================================
   lib/upload.js — XHR-based upload with per-file progress + cancellation.
   Preserves the original state machine exactly:
     queued → uploading → indexing → done | failed | cancelled
   Abort is frontend-only (A-RESOLVED-10): backend keeps processing whatever
   bytes already arrived. This is intentional.
   ========================================================================== */
(function (global) {
  "use strict";

  const { ref } = Vue;

  function useUpload() {
    // Singleton: multiple call sites in different components must share
    // the same reactive refs, otherwise docs/uploadProgress diverge.
    if (global.__useUploadSingleton) return global.__useUploadSingleton;

    const docs = ref([]);
    const uploadProgress = ref({});   // filename → { loaded, total, phase, error }
    const uploadXhrs = ref({});       // filename → XMLHttpRequest
    const uploadResults = ref([]);    // recent terminal results for the log

    async function loadDocs() {
      try {
        const j = await global.api("/api/documents");
        docs.value = j.documents || [];
      } catch (e) {
        global.ElementPlus.ElMessage.error(e.message);
      }
    }

    /**
     * Element Plus upload hook. Element Plus calls this with `{ file, onSuccess,
     * onError, ... }`. We bypass its built-in fetch in favour of XHR so we
     * get real byte-level progress events.
     */
    function onUpload(opt) {
      const f = opt.file;
      const fd = new FormData();
      fd.append("file", f);

      const xhr = new XMLHttpRequest();
      uploadXhrs.value[f.name] = xhr;
      // Initialise progress entry so the UI can show 0% immediately. The
      // spread + overwrite pattern preserves any future fields cleanly.
      uploadProgress.value[f.name] = {
        loaded: 0, total: f.size, phase: "queued", error: null,
      };

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          uploadProgress.value[f.name] = {
            ...uploadProgress.value[f.name],
            loaded: e.loaded, total: e.total, phase: "uploading",
          };
        }
      };

      // Bytes are fully sent. Backend now indexes synchronously; we can't see
      // per-page progress (per DD1-A), so we hold at 99% and show an
      // indeterminate spinner in the "indexing" phase.
      xhr.upload.onload = () => {
        uploadProgress.value[f.name] = {
          ...uploadProgress.value[f.name],
          loaded: uploadProgress.value[f.name].total,
          phase: "indexing",
        };
      };

      xhr.onload = () => {
        try {
          const j = JSON.parse(xhr.responseText);
          const ok = (j.succeeded || 0) > 0;
          uploadProgress.value[f.name] = {
            ...uploadProgress.value[f.name],
            phase: ok ? "done" : "failed",
            error: ok ? null : (j.error || "unknown"),
          };
          // Prominent failure notification when any file in this response
          // failed to index (e.g. revoked LLM key → HTTP 200 with
          // succeeded:0). Mirrors prior behaviour.
          const okCount = (j.succeeded || 0);
          const totalCount = (j.total != null ? j.total : (j.results || []).length);
          if (okCount < totalCount) {
            const failedCount = totalCount - okCount;
            let firstErr = (j.results || []).find(r => !r.success)?.error || "";
            if (firstErr.length > 120) firstErr = firstErr.slice(0, 120) + "…";
            const message = totalCount <= 1
              ? f.name + (firstErr ? "：" + firstErr : "")
              : failedCount + "/" + totalCount + " 个文档索引失败" + (firstErr ? "：" + firstErr : "");
            global.ElementPlus.ElNotification({
              title: "文档索引失败", type: "error", duration: 6000,
              position: "top-right", message,
            });
          }
          uploadResults.value.push({ name: f.name, ok, raw: j });
          if (ok) loadDocs();
          ok ? opt.onSuccess() : opt.onError();
        } catch (e) {
          uploadProgress.value[f.name] = {
            ...uploadProgress.value[f.name],
            phase: "failed", error: e.message,
          };
          global.ElementPlus.ElMessage.error(e.message);
          opt.onError();
        } finally {
          delete uploadXhrs.value[f.name];
        }
      };

      xhr.onerror = () => {
        uploadProgress.value[f.name] = {
          ...uploadProgress.value[f.name],
          phase: "failed", error: "network error",
        };
        global.ElementPlus.ElMessage.error("网络错误");
        delete uploadXhrs.value[f.name];
        opt.onError();
      };

      // Abort is frontend-only per A-RESOLVED-10: backend doesn't see the
      // abort and will keep indexing whatever bytes already arrived.
      xhr.onabort = () => {
        uploadProgress.value[f.name] = {
          ...uploadProgress.value[f.name],
          phase: "cancelled", error: null,
        };
        delete uploadXhrs.value[f.name];
      };

      const h = {};
      if (global.apiKey()) h["X-API-Key"] = global.apiKey();
      xhr.open("POST", "/upload");
      Object.entries(h).forEach(([k, v]) => xhr.setRequestHeader(k, v));
      xhr.send(fd);
    }

    function cancelUpload(filename) {
      const xhr = uploadXhrs.value[filename];
      if (xhr) xhr.abort();
    }

    async function deleteDoc(doc) {
      try {
        await global.ElementPlus.ElMessageBox.confirm(
          `删除文档 "${doc.doc_name || doc.pdf_name || doc.id}"?此操作不可恢复`,
          "确认删除",
          {
            confirmButtonText: "删除",
            cancelButtonText: "取消",
            type: "warning",
            confirmButtonClass: "el-button--danger",
          }
        );
      } catch (_) {
        return; // user cancelled — ElMessageBox rejects on cancel
      }
      try {
        await global.api(`/api/documents/${doc.id}`, { method: "DELETE" });
        global.ElementPlus.ElMessage.success("文档已删除");
        await loadDocs();
      } catch (e) {
        global.ElementPlus.ElMessage.error(`删除失败: ${e.message}`);
      }
    }

    return global.__useUploadSingleton = {
      docs, uploadProgress, uploadXhrs, uploadResults,
      loadDocs, onUpload, cancelUpload, deleteDoc,
    };
  }

  Object.assign(global, { useUpload });
})(typeof window !== "undefined" ? window : globalThis);
