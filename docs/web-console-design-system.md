# Web Console Design System

Source of truth: [`web/static/tokens.css`](../web/static/tokens.css).

The web console uses a **hybrid** design language — **C**-school (editorial calm: Aesop, MUJI, paper-like) as the substrate, **A**-school (builder SaaS precision: Linear, Vercel, JetBrains-Mono-for-data) for the details.

## Token catalog

### Color — single-accent, paper-warm

| Token | Value | Usage |
|---|---|---|
| `--bg` | `#F7F2E7` | Page background (warm cream paper) |
| `--bg-2` | `#F0E9DA` | Subtle paper texture / inset wells |
| `--surface` | `#FFFFFF` | Primary card |
| `--surface-2` | `#FBF7EC` | Raised inset, document-like |
| `--surface-3` | `#F2ECDB` | Hover on surface-2 |
| `--ink` | `#1A1A1A` | Primary text |
| `--ink-2` | `#3F3A33` | Secondary text |
| `--ink-muted` | `#7A7166` | Captions, labels, eyebrow |
| `--ink-faint` | `#ADA395` | Disabled, very light hints |
| `--accent` | `#2E3D6B` | **The** accent — deep ink blue |
| `--accent-2` | `#3F5290` | Accent hover |
| `--accent-soft` | `rgba(46,61,107,0.08)` | Soft accent tint |
| `--accent-line` | `rgba(46,61,107,0.20)` | Hairline accent border |
| `--success` | `#5C8A6E` | Muted sage (done state) |
| `--warn` | `#B8884A` | Ochre (cancelled state) |
| `--danger` | `#B84A3E` | Brick red (failed state) |
| `--hairline` | `#E5DED0` | 1px warm-gray border |
| `--hairline-2` | `#D6CDB9` | Stronger hairline |

### Type — three families

| Role | Family | Usage |
|---|---|---|
| Display | **Fraunces** (variable serif, opsz 9–144) | Brand wordmark, panel titles (26px / `opsz: 144` for tighter display glyphs) |
| UI | **Inter** (400/500/600) | Body, labels, buttons, table cells |
| Mono | **JetBrains Mono** (400/500) | Paths, IDs, byte sizes, status chips, percentage, key icon |

Loaded from Google Fonts in `index.html` with `preconnect` to cut the cross-origin handshake.

### Spacing — 4px base

```
--s-1:  4px     --s-5: 20px
--s-2:  8px     --s-6: 24px
--s-3: 12px     --s-7: 32px
--s-4: 16px     --s-8: 40px
                    --s-9: 56px
```

### Radius — hierarchical

```
--r-xs:   6px     small (chips, code)
--r-sm:  10px     inputs, buttons, panels
--r-md:  14px     cards
--r-lg:  18px     outer panel
--r-pill: 999px   chips, status badges
```

### Shadow — paper-soft, layered

```
--shadow-1: 0 1px 2px rgba(40,30,15,.04)
--shadow-2: 0 1px 2px rgba(40,30,15,.04), 0 4px 14px rgba(40,30,15,.06)
--shadow-3: 0 1px 2px rgba(40,30,15,.05), 0 8px 28px rgba(40,30,15,.08)
```

### Motion — C's breath + A's precision

```
--ease:       cubic-bezier(0.4, 0, 0.2, 1)   micro
--ease-out:   cubic-bezier(0.2, 0.7, 0.2, 1) panel
--dur-micro:  120ms   button hover, input focus
--dur-panel:  240ms   answer card rise, evidence expand
--dur-page:   360ms   (reserved for future page-level transitions)
```

---

## Component inventory

| Component | Source | Notes |
|---|---|---|
| **Topbar** | `components/topbar.js` | Brand mark (3×3 SVG grid) + API key input + save button + status dot |
| **TabNav** | `components/tab-nav.js` | Element Plus `<el-tabs>` wrapper, slots for the 3 panels |
| **Panel** | `components/panel.js` | Reusable shell: eyebrow + title + description + actions slot + default slot |
| **DocsPanel** | `components/docs-panel.js` | Upload dropzone + per-file progress list + doc table |
| **QAPanel** | `components/qa-panel.js` | Ask input + answer card (loading skeleton / answer / empty hint) + evidence chain |
| **ConfigPanel** | `components/config-panel.js` | "Current" snapshot + form + persist toggle + connectivity self-test |
| **ConfidenceMeter** | `components/confidence-meter.js` | Label → meter → value chip, mapped from `answer.confidence` enum |
| **UploadProgressList** | `components/upload-progress-list.js` | Per-file progress rows with state machine: queued → uploading → indexing → done / failed / cancelled |
| **CurrentConfigCard** | `components/current-config-card.js` | "当前" snapshot: model · provider · base_url · key status |

---

## Patterns

### Three-step type hierarchy (every panel)

```
.section-eyebrow   — Inter 11px uppercase letterspacing 0.8px, ink-muted
.section-title     — Fraunces 26px / opsz 144, ink
.section-desc      — Inter 13px, ink-muted
```

This rhythm makes the page scan-able: eyebrow tags the domain, title states the action, description sets expectations.

### Status chips (state communication)

Mono + uppercase + pill + 1px semantic border + soft tint:

```
.chip.chip--accent   ← uploading / indexing / queued
.chip.chip--ok       ← done
.chip.chip--warn     ← cancelled
.chip.chip--danger   ← failed
```

Used in upload rows, evidence-chain headers, config alerts. The leading `::before` dot uses `currentColor` at 55% opacity — gives the chip a heartbeat without colour noise.

### Evidence chain (RAG result presentation)

```
Evidence · 12 篇 · 8 节点 · 24 页
[per-doc collapse: 节点 (4) ── 页码片段 (3)]
```

Each match is a collapsible `el-collapse-item` with two sections: selected nodes (title + path + summary + page numbers) and page snippets (mono page-number chip + text). Defense-in-depth: missing `doc_id` falls back to `pdf_name`; legacy nodes without docs attach to the first match so they're never silently dropped.

### Confidence meter

Driven by `answer.confidence` enum (high / medium / low / unknown) — **never fabricates a percentage**:

```
ratio   = high→100%, medium→66%, low→33%, unknown→0%
label   = 高 / 中 / 低 / 未知
cssClass = .conf-val.high|medium|low
```

The bar uses a horizontal gradient `var(--warn) → var(--success)`; the value chip carries the semantic class so the bar and chip agree on tone.

### XHR upload state machine

```
queued ──bytes sent──▶ uploading ──bytes done──▶ indexing ──response──▶ done
                                            │
                                            └──(partial / total mismatch)──▶ failed
                                                              │
                                                              └──(abort)──▶ cancelled
```

Frontend-only abort per `A-RESOLVED-10` — backend keeps processing whatever bytes already arrived. Toast for partial-failure cases (e.g. revoked LLM key → HTTP 200 with `succeeded:0`).

### Save-the-key + auto-refetch (web ↔ server)

```
user types key → 保存
  ├─ localStorage["pageindex_api_key"] = v
  ├─ hasKey.value = Boolean(v)   (triggers keyInput watcher)
  ├─ ElementPlus.ElMessage.success("Key 已保存")
  └─ if v: useUpload().loadDocs() + useConfig().loadConfig()
```

The auto-refetch means the user sees the docs list populate without a manual page reload. The original behaviour (mount-only fetch) was the RC2 bug.

---

## What to NOT do

These are the patterns that were explicitly rejected — calling them out so they don't creep back in:

- ❌ **Purple-pink-blue gradients** on hero sections — every AI-era SaaS has this, it reads as "AI demo page" not "designed product".
- ❌ **Inter / Roboto / system-ui as display** — Inter is fine for UI; never for the headline.
- ❌ **Rounded card + colored left-border accent** — Material-era leftover, now visual noise on dashboards.
- ❌ **Emoji as icon substitute** (🚀 ⚡ ✨) — the brand mark is SVG; upload arrow is SVG stroke; status is mono chip.
- ❌ **Fabricated stats / fake logos / dummy testimonials** — placeholders must signal "real data needed", never invent it.
- ❌ **`fetch` then forget** — every network call must either update reactive state, or surface a toast on failure. Silent failures are bugs.