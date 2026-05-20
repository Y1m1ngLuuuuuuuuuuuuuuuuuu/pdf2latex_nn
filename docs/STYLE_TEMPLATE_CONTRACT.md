# Style Template Contract

**Last updated**: 2026-05-20

This contract defines how layout style, journal templates, and renderer presets
enter the generator. The boundary object is `StyleProfile`.

## Owner Modules

```text
src/generation/style_profile.py
src/generation/render_surface.py
src/generation/ir_renderer.py
src/generation/ir_renderers/
src/ir/schema.py
```

`StyleProfileExtractor` infers an original-like profile from `DocumentIR`.
Future template engines should produce or transform `StyleProfile` instead of
patching renderer internals.

## Boundary Object

`StyleProfile` owns renderer-facing style decisions:

| Field | Responsibility |
| --- | --- |
| `documentclass` / `document_class` | LaTeX document class. |
| `class_options` | Options such as `twocolumn`, font size, paper size. |
| `packages` | Required LaTeX packages. |
| `macros` / `preamble_lines` | Template macros and spacing helpers. |
| `page_layout` | Paper size, margins, columns, gutter, body font. |
| `column_mode` | `single`, `two_column`, or `mixed`. |
| `role_styles` | Heading, caption, abstract, reference, footnote styling. |
| `renderer_options` | Safe renderer switches. |
| `bibliography_style` | Reference layout hints. |
| `header_footer` | Page furniture reproduction hints. |

The renderer consumes `StyleProfile`; it should not independently infer global
style when a profile field already exists.

## Modes

The current system supports three conceptual modes:

| Mode | Goal |
| --- | --- |
| `original_like` | Infer the visible source style from PDF geometry and spans. |
| `template` | Use a named template such as IEEE, ACM, SCI-like, or journal-specific. |
| `hybrid` | Keep original content geometry but normalize to a target template. |

Only `original_like` is the current production default. Template mode should be
implemented as a `StyleProfile` provider, not as another renderer path.

## Template Provider Contract

A future provider should expose:

```python
class StyleTemplateProvider:
    def build_profile(self, document: DocumentIR, *, template_name: str, mode: str) -> StyleProfile:
        ...
```

It may use:

- document page size;
- body font estimate;
- title/authors/abstract geometry;
- detected column mode;
- user-selected template name;
- optional journal metadata.

It must not use:

- TeX source during inference;
- GNN view records as the complete document;
- gold labels or evaluation results.

## Column Contract

Column decisions must be derived from body text, not front matter alone.

Rules:

1. Author blocks and affiliations may be multi-column but must not force the
   whole document into two-column mode.
2. Abstract is rendered as a separate semantic region but still follows the
   page/body column strategy when the original does.
3. References and appendix can use their own local column policy based on bbox
   width and section-tail evidence.
4. Only cross-column floats, wide equations, or explicit mixed-column bands may
   trigger local single-column transitions.

## Heading Style Contract

Heading text and numbering are separated:

```text
visible heading text
numbering style
render level
starred/unstarred decision
```

The renderer may let LaTeX own numbering only when the visible numbering matches
the default counter for the chosen level. Non-default prefixes such as `0.1`,
Roman/alpha custom styles, and Chinese numbering should preserve visible text
with starred commands unless a template explicitly owns that numbering.

Virtual heading nodes are decoder-side. They may enter `RenderTreeIR`, but they
do not modify v7 or graph tensors.

## Float Style Contract

Figure/table/algorithm rendering uses float slots:

```text
float visual bbox -> width decision
caption group     -> \caption and \label
asset/crop/table  -> body renderer
```

The style template may change float placement defaults, but production
reconstruction uses `[H]` or constrained placement when visual order is the
priority. Template mode may relax that only when explicitly requested.

## Extension Points

Safe extension points:

- add a new `StyleProfile` field with default;
- add a template provider that returns `StyleProfile`;
- add role style keys consumed by `ir_renderers`;
- add package/macro lines through profile, not hardcoded renderer globals.

Unsafe changes:

- adding another production renderer path;
- making old `latex_renderer.py` a competing surface;
- deriving global style from GNN view only;
- changing graph feature schema for a renderer-only template.

## Validation

For renderer/template changes, run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_ir_renderer_registry.py \
  tests/test_generation_style_citations.py \
  tests/test_postprocess_renderer.py
```

For style extraction changes, also run:

```bash
.venv/bin/python -m pytest -q tests/test_mineru_v7_document_ir_adapter.py
```

