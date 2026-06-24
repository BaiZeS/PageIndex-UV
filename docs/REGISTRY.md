# docs REGISTRY.md

> 文档注册表（Document Consistency Protocol v1.8.0）。每次写/改文档更新此处 Last Modified + Depends On。
> 既有 design-docs（db-concurrency / non-vector-retrieval / project-refactor / batch-upload / super-tree-retrieval-v3 / delete-path-integrity / model-config-completion）为历史产出，尚未纳入本表跟踪，仅新文档登记。

| 文档路径 | 类型 | Last Modified | Depends On | 状态 |
|---------|------|--------------|-----------|------|
| docs/design-docs/PageIndex/web-console/spec.md | spec | 2026-06-24 | — | CURRENT |
| docs/design-docs/PageIndex/web-console/tasks.md | tasks | 2026-06-24 | spec.md | CURRENT |

> staleness 规则：读文档前，若其 Depends On 文档的 Last Modified 比自身新 → STALE，先更新依赖再据此决策。
