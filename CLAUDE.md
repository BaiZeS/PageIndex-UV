# CLAUDE.md

本项目的 workflow 与编码规则定义在 [AGENTS.md](./AGENTS.md)，**全部规则同等适用于 Claude Code**。

每次会话开始与执行任何代码相关任务前，请先阅读并严格遵循 AGENTS.md 中的：

1. **强制规则** —— 代码变更统一入口（必先加载 `workflow-code-generation`）
2. **Workflow 流水线** —— 按场景路由到对应 skill
3. **默认 Agents/Skills** —— 在 Claude Code 中通过 Skill / Agent 工具以**同名**方式调用，无需手动配置
4. **设计文档路径** —— `docs/design-docs/<module>/<feature>/{spec,tasks}.md`
5. **例外情况** —— 纯信息查询、非代码文件修改、用户显式声明跳过

> AGENTS.md 中所述 `~/.codex/agents/` 路径为 Codex 配置位置；Claude Code 使用 Agent 工具内置的同名 subagent_type（`codebase-researcher` / `performance-reviewer` / `robustness-reviewer` / `standards-reviewer` / `magical-prompt-reviewer` / `spec-compliance-reviewer` / `review-critic`），无需额外初始化。

## 单一信息源

修改工作流规则时**只改 AGENTS.md**，本文件保持为指针，避免双份维护漂移。
