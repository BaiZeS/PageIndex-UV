# AGENTS.md — Workflow 默认启用

本文件配置 Codex 在此仓库中默认启用 workflow 系列 skills 和 agents。

## 强制规则

### 代码变更统一入口

**所有代码文件（.cc/.cpp/.h/.go/.py 等）的变更请求——包括新功能、Bug 修复、重构、优化——必须首先加载 `workflow-code-generation` skill**，按其工作流程执行。

非代码文件（.md 等）的修改不需要走 workflow。

### Workflow 流水线

代码变更请求按以下优先级自动路由：

| 场景 | 自动加载的 Skill |
|------|-----------------|
| 需求不明确、无 spec.md | `workflow-requirements-clarification` → `workflow-system-design` → `workflow-code-generation` |
| 有 spec.md 但设计章节为空 | `workflow-system-design` → `workflow-code-generation` |
| spec.md + tasks.md 完整 | `workflow-code-generation`（内部自动触发 `workflow-code-review`） |
| 用户要求写测试 | `workflow-test-generation` |
| 用户要求 Code Review | `workflow-code-review` |

### 默认启用的 Agents

以下 agents 已在 `~/.codex/agents/` 中配置，workflow 执行时自动调用：

| Agent | 用途 | 调用方式 |
|-------|------|----------|
| `codebase-researcher` | 深度代码调研 | `workflow-requirements-clarification` 和 `workflow-system-design` 自动调用 |
| `performance-reviewer` | 性能审查 | `workflow-code-review` 自动并行调用 |
| `robustness-reviewer` | 健壮性审查 | `workflow-code-review` 自动并行调用 |
| `standards-reviewer` | 工程规范审查 | `workflow-code-review` 自动并行调用 |
| `magical-prompt-reviewer` | 契约与信任链审查 | `workflow-code-review` 自动并行调用 |
| `spec-compliance-reviewer` | 需求/设计符合度审查 | `workflow-code-review` 自动并行调用 |
| `review-critic` | 争议问题对抗性验证 | `workflow-code-review` 有 finding 时自动调用 |

### 默认启用的编码规范 Skills

以下 skills 在编码阶段自动加载：

| Skill | 加载条件 |
|-------|---------|
| `bp-coding-best-practices` | 始终加载 |
| `bp-performance-optimization` | 始终加载 |
| `bp-architecture-design` | 系统设计阶段 |
| `bp-component-design` | 组件设计阶段 |
| `bp-distributed-systems` | 涉及网络通信、多节点协调时 |
| `std-cpp` | C++ 文件 |
| `std-go` | Go 文件 |

## 设计文档路径

- spec.md / tasks.md 一律存放在 `docs/design-docs/<module>/<feature>/` 下
- 文件名必须是 `spec.md` 和 `tasks.md`

## 例外

以下场景不走 workflow：
- 纯信息查询（"这个函数做什么？"）
- 非代码文件的修改
- 用户明确说"快速修改，不走 workflow"
