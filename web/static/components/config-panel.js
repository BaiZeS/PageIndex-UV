/* ============================================================================
   components/config-panel.js — Model config tab: current snapshot + form +
   connectivity self-test. T13 (FR12) test alert binds to testResult.
   ========================================================================== */
(function (global) {
  "use strict";

  const { computed } = Vue;

  const ConfigPanel = {
    name: "ConfigPanel",
    setup() {
      const { config, cfgForm, cfgMsg, testResult, saveConfig, testConnectivity } = global.useConfig();

      // Pre-fill derived values from cfgForm so the alert is reactive.
      const testTitle = computed(() => {
        const t = testResult.value;
        if (!t || t.status === "loading") return "测试中…";
        if (t.ok) {
          const parts = ["✓ 连接成功"];
          if (t.latency_ms != null) parts.push(`(${t.latency_ms}ms`);
          if (t.model) parts.push(` · ${t.model}`);
          if (t.latency_ms != null) parts.push(")");
          return parts.join("");
        }
        return `✗ ${t.error || 'unknown'}${t.detail ? ' · ' + t.detail : ''}`;
      });
      const testType = computed(() => {
        const t = testResult.value;
        if (!t || t.status === "loading") return "info";
        return t.ok ? "success" : "error";
      });

      return { config, cfgForm, cfgMsg, testResult, saveConfig, testConnectivity, testTitle, testType };
    },
    template: `
      <panel
        title="模型配置"
        eyebrow="Configuration"
        desc="当前值已填入下方（来自上方「当前」快照）。修改后点击「应用」即生效；清空某项表示保持不变。"
        :narrow="false">

        <current-config-card :config="config" />

        <el-form label-width="120px" class="cfg-form">
          <el-form-item label="模型名">
            <el-input v-model="cfgForm.model" placeholder="未设置"></el-input>
          </el-form-item>
          <el-form-item label="检索模型">
            <el-input v-model="cfgForm.retrieve_model" placeholder="未设置"></el-input>
          </el-form-item>
          <el-form-item label="API Key">
            <el-input v-model="cfgForm.api_key" show-password placeholder="留空表示不修改"></el-input>
          </el-form-item>
          <el-form-item label="Base URL">
            <el-input v-model="cfgForm.base_url" placeholder="未设置"></el-input>
          </el-form-item>
          <el-form-item label="持久化">
            <el-switch v-model="cfgForm.persist"></el-switch>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="saveConfig">应用</el-button>
            <span v-if="cfgMsg" class="cfg-msg">{{ cfgMsg }}</span>
          </el-form-item>

          <!-- T13 (FR12): connectivity self-test -->
          <el-form-item label="连通性测试">
            <el-button
              class="test-btn"
              :loading="testResult && testResult.status === 'loading'"
              @click="testConnectivity">测试当前配置</el-button>
          </el-form-item>
          <el-form-item v-if="testResult && testResult.status !== 'loading'" class="test-result">
            <el-alert
              :type="testType"
              :title="testTitle"
              :closable="false"
              show-icon />
          </el-form-item>
        </el-form>
      </panel>
    `,
  };

  Object.assign(global, { ConfigPanel });
})(typeof window !== "undefined" ? window : globalThis);
