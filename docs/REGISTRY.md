# docs REGISTRY.md

> 文档注册表（Document Consistency Protocol v1.8.0）。每次写/改文档更新此处 Last Modified + Depends On。
> 既有 design-docs（db-concurrency / non-vector-retrieval / project-refactor / batch-upload / super-tree-retrieval-v3 / delete-path-integrity / model-config-completion）为历史产出，尚未纳入本表跟踪，仅新文档登记。

| 文档路径 | 类型 | Last Modified | Depends On | 状态 |
|---------|------|--------------|-----------|------|
| docs/design-docs/PageIndex/web-console/spec.md | spec | 2026-06-26 | — | CURRENT |
| docs/design-docs/PageIndex/web-console/tasks.md | tasks | 2026-06-26 | spec.md | CURRENT |

> 变更日志：
> - 2026-06-26：`web-console/spec.md` 由 v1.0（FR1-FR8）扩展为 v1.1（+FR9-FR12：上传进度条、文档删除、问答证据、模型连通性测试）；§4-8 待 system-design 阶段产出。
> - 2026-06-26：`web-console/tasks.md` 标记 STALE（Depends On 的 spec.md 已被扩展，需在 v1.1 task-planning 阶段更新为 v1.1 tasks）。本轮 LEAF EXECUTOR 范围不含更新 tasks.md。
> - 2026-06-26：`web-console/spec.md` §4-8 由 design 阶段产出（DD1=A 前端模拟/DD2=A 单次确认/DD3=A Accordion/DD4=A chat ping；A5-A10 全部 RESOLVED；新增 `DELETE /api/documents/{id}` + `POST /api/config/test`；前端扩展 uploadProgress/deleteDoc/evidenceModule/testConnectivity）；spec.md=CURRENT。`tasks.md` 仍 STALE（task-planning 阶段更新为 v1.1 tasks）。
> - 2026-06-26：`web-console/tasks.md` 由 v1.0（9 任务 DONE，commit 5b30adc）扩展为 v1.1（T1-T14 共 14 任务）：保留 T1-T9 为 DONE；新增 T10-T14 覆盖 FR9-FR12（T10=DELETE endpoint/T11=POST /api/config/test/T12=onUpload XHR 改写/T13=deleteDoc+evidenceModule+testConnectivity UI/T14=README+回归）；依赖图 T10∥T11∥T12→T13→T14；atomicity 每任务 ≤3 文件 ≤5 分钟；后端 TDD（RED→GREEN），前端 browser-testing 手测。`tasks.md` = CURRENT。

> staleness 规则：读文档前，若其 Depends On 文档的 Last Modified 比自身新 → STALE，先更新依赖再据此决策。
