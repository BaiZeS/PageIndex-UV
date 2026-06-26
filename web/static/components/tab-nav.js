/* ============================================================================
   components/tab-nav.js — Three-tab nav: 文档 / 问答 / 模型配置.
   Thin wrapper around Element Plus el-tabs that supports slot-in panels
   (Element Plus's default slot content lives inside each el-tab-pane).
   ========================================================================== */
(function (global) {
  "use strict";

  const { computed } = Vue;

  const TabNav = {
    name: "TabNav",
    props: {
      modelValue: { type: String, required: true },
    },
    emits: ["update:modelValue"],
    setup(props, { emit }) {
      const activeTab = computed({
        get: () => props.modelValue,
        set: (v) => emit("update:modelValue", v),
      });
      return { activeTab };
    },
    template: `
      <el-tabs v-model="activeTab" class="console-tabs">
        <el-tab-pane label="文档" name="docs">
          <slot name="docs" />
        </el-tab-pane>
        <el-tab-pane label="问答" name="qa">
          <slot name="qa" />
        </el-tab-pane>
        <el-tab-pane label="模型配置" name="cfg">
          <slot name="cfg" />
        </el-tab-pane>
      </el-tabs>
    `,
  };

  Object.assign(global, { TabNav });
})(typeof window !== "undefined" ? window : globalThis);
