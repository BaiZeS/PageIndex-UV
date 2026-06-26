/* ============================================================================
   components/qa-panel.js — Q&A tab: ask input + answer card + evidence chain.
   ========================================================================== */
(function (global) {
  "use strict";

  const QAPanel = {
    name: "QAPanel",
    setup() {
      // docs ref needed only for matchedRows fallback (defense-in-depth join
      // against the loaded docs list). Currently unused in render but kept
      // to mirror the original behaviour if a future matched-row list
      // re-introduces it.
      global.useUpload();  // singleton side-effect — ensures docs ref exists
      const { query, answer, searching, doSearch } = global.useQA();

      return { query, answer, searching, doSearch, evidenceModule: global.evidenceModule };
    },
    template: `
      <panel
        title="检索问答"
        eyebrow="Ask"
        desc="对已索引文档提问，返回答案与命中的文档片段。">
        <div class="ask-row">
          <el-input
            v-model="query"
            type="textarea"
            :rows="3"
            placeholder="输入问题，例如：这份文档说明了什么？"
            class="ask-input"
            resize="none"
            @keydown.meta.enter="doSearch"
            @keydown.ctrl.enter="doSearch"></el-input>
          <el-button
            type="primary"
            :loading="searching"
            @click="doSearch"
            class="ask-btn">
            {{ searching ? '检索中…' : '提问' }}
          </el-button>
        </div>

        <div v-if="searching" class="answer-card loading">
          <div class="skel-line w-90"></div>
          <div class="skel-line w-80"></div>
          <div class="skel-line w-60"></div>
          <div class="skel-line w-40"></div>
        </div>

        <article v-else-if="answer" class="answer-card">
          <confidence-meter :confidence="answer.confidence" />
          <p class="answer-text">{{ answer.answer }}</p>

          <!-- T13 (FR11): evidence chain — accordion of matched docs with
               selected_nodes + page snippets. -->
          <component :is="evidenceModule(answer)" v-if="answer" />
        </article>

        <div v-else class="empty-hint">输入问题后点击「提问」查看检索结果。</div>
      </panel>
    `,
  };

  Object.assign(global, { QAPanel });
})(typeof window !== "undefined" ? window : globalThis);
