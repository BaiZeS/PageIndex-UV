/* ============================================================================
   components/docs-panel.js — Documents tab: drop-zone, progress list, table.
   Wires the upload + docs composable to the presentational primitives.
   ========================================================================== */
(function (global) {
  "use strict";

  const { computed } = Vue;

  const DocsPanel = {
    name: "DocsPanel",
    setup() {
      const {
        docs, uploadProgress, uploadResults,
        loadDocs, onUpload, cancelUpload, deleteDoc,
      } = global.useUpload();

      const uploadingCount = computed(() => {
        return Object.values(uploadProgress.value || {})
          .filter(p => p.phase === "queued" || p.phase === "uploading" || p.phase === "indexing")
          .length;
      });

      return {
        docs, uploadProgress, uploadResults, uploadingCount,
        loadDocs, onUpload, cancelUpload, deleteDoc,
      };
    },
    template: `
      <panel
        title="文档库"
        eyebrow="Documents"
        desc="已索引的文档。拖拽 PDF / Markdown 到下方上传。">
        <template #actions>
          <el-button size="small" class="ghost-btn" @click="loadDocs">
            <span class="ico">↻</span> 刷新
          </el-button>
        </template>

        <el-upload
          :http-request="onUpload"
          drag
          multiple
          :show-file-list="false"
          class="doc-uploader">
          <div class="uploader-body">
            <div class="uploader-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
                   stroke="currentColor" stroke-width="1.6"
                   stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 16V5"/>
                <path d="M7 10l5-5 5 5"/>
                <path d="M4 20h16"/>
              </svg>
            </div>
            <div class="uploader-title">拖拽或点击上传</div>
            <div class="uploader-hint">PDF · Markdown</div>
          </div>
        </el-upload>

        <upload-progress-list
          :progress-map="uploadProgress"
          @cancel="cancelUpload" />

        <div class="doc-table-wrap">
          <el-table :data="docs" class="doc-table" :empty-text="'暂无文档 — 上传后这里会出现列表'">
            <el-table-column prop="id" label="#" width="64" align="center"></el-table-column>
            <el-table-column prop="doc_name" label="文档" min-width="180"></el-table-column>
            <el-table-column prop="doc_description" label="描述" min-width="240" show-overflow-tooltip></el-table-column>
            <el-table-column label="操作" width="100" align="center">
              <template #default="scope">
                <el-button size="small" type="danger" plain @click="deleteDoc(scope.row)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <ul v-if="uploadResults.length" class="upload-log">
          <li v-for="(r, i) in uploadResults" :key="i" :class="r.ok ? 'ok' : 'fail'">
            <span class="log-status">{{ r.ok ? '✓' : '✕' }}</span>
            <span class="log-name">{{ r.name }}</span>
          </li>
        </ul>
      </panel>
    `,
  };

  Object.assign(global, { DocsPanel });
})(typeof window !== "undefined" ? window : globalThis);
