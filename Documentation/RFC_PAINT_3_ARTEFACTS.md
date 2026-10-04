# RFC PAINT-3 — `plakat paint` artefacts: the ANALYSIS and the OUTCOME sheet

Status: **PROPOSED** (plan only). Depends on RFC PAINT-1 (the stroke score, the pass dumps).

## 1. Summary

Besides the picture, a paint run can emit two artefacts, each switched on by a CLI flag or a plan key:

```
plakat paint from pic.png --plan plan.hjson -o out.png \
    --analysis out.md          # plan key  analysis: true   (or a path)
    --outcome  out_sheet.png   # plan key  outcome:  true   (or a path; .png / .pdf)
```

- **analysis** — a Markdown report of every parameter the run used, each with the comment that explains
  it, plus what the run FOUND (faces, subject matte, hair, the budget, the pass schedule with counts and
  times, the palette) and what it MEASURED (height per plane, fine-stroke lengths, dry share, flow).
- **outcome** — a one-page sheet in the format of the sample (`paint_analysis/Vites_23_analysis.jpeg`):
  the master composition, four micro-analysis insets, the layer hierarchy, the palette, the brushwork
  concept. Every panel is derived from the run; nothing is invented.

Principle: **the sheet reports, it does not flatter**. A panel that cannot be backed by a measurement is
left out rather than decorated.

## 2. The analysis (Markdown)

Sections, in order:

1. **Run** — source file, size, medium, seed, version, wall time, the command line.
2. **Plan** — the plan file verbatim WITH its comments (they are the art director's reasoning), then the
   resolved value of every key the painter read (plan ∪ CLI ∪ defaults), marked `plan` / `cli` / `default`.
3. **What the run found** — face count and extent, subject matte %, semantic regions, hair/fur extent,
   HDR applied, budget and its formula, armature sides per tier, reserve/protect decisions.
4. **The passes** — the table the terminal prints (stage, radius, strokes, rate, seconds) plus per pass:
   mean stroke length/width, dry share (wet < 0.62), face-only, flow strength.
5. **Measurements** — from the finished canvas: height per plane (face / subject / background / lights /
   shadows, % of peak), weave footprint, fine-stroke length histogram, pigment usage ranking.
6. **Dials glossary** — for each non-default dial, its PAINT_CONTROLS row (so the reader learns what
   `ridges 0.8` meant without opening the docs).

All of this exists in the painter today as `PaintResult.stats`, the score header, the masks, the height
field (`PLAKAT_PAINT_HEIGHT_DUMP`) and the plan's own comments; the work is a `report.rs` that gathers
them and writes Markdown.

## 2b. Insights and recommendations (LLM, optional) — P0.5

When prompt enrichment is configured (a hosted provider or Ollama), the analysis can be run THROUGH the
LLM to add two sections the facts alone cannot write: **Insights** (what the numbers say about the
painting) and **Recommendations** (which dials to move, by how much, and why — as a plan diff the user
can paste).

```
--analysis-insights          analysis_insights: true     (requires --analysis / analysis: true)
```

Design rules:
- **Input = the facts.** The model receives the analysis Markdown (plan with comments, resolved dials,
  findings, passes, measurements, pigment usage) and the PAINT_CONTROLS glossary for the medium. It does
  NOT receive the picture unless the provider is a vision model and `--analysis-insights=with-image` is
  given (a downsized copy; then the model may also judge what it sees, and says so).
- **System prompt** (the whole value is here; kept in `assets/prompts/paint_insights.md`, versioned):
  the model is a painting technician reading a run report; it reasons ONLY from the numbers given and
  cites the measurement behind every claim ("height on faces 9.5 vs beard 17 → …"); it recommends dials
  from the glossary ONLY, with a value and a one-line reason each, as an hjson snippet; it must say what
  it cannot know from the facts (anything about likeness, taste, the subject); it never invents
  techniques or pigment names; it keeps the user's hard rules (faces recognizable; no scene-specific
  defaults; one heavy job at a time is not its concern). Medium-specific sections (oil: thickness, relief,
  weave, sheen; watercolour: reserve, flow, grain, dry brush, fine lines).
- **Output** appended to the same `.md` under `## Insights` and `## Recommendations`, each
  recommendation as `key: value   # reason` so the user can copy it into the plan; plus a line naming the
  model and provider used.
- **Privacy**: a hosted provider receives the report text (and the image when asked) — it leaves the
  machine; Ollama keeps it local. The docs and the terminal line say which.
- **Loop**: `plakat paint … --analysis --analysis-insights` → read the recommendations → edit the plan →
  rerun. Later (P3) `--analysis-apply` could write the recommended plan as `plan.next.hjson` for review,
  never overwriting the user's plan.

## 3. The outcome sheet

### 3.1 Panels and their sources

| panel | source (facts) | how it is chosen / labelled |
|---|---|---|
| **Master composition** | the render | — |
| **Micro-analysis 1–4** | four 1:1 crops of the render (circular "lens" crops) | §3.2 |
| **Layer hierarchy** | the per-pass canvas dumps (the painter already writes them under `PLAKAT_WCB_DUMP`; make that a first-class option) | drawn as an isometric stack: block-in → restates → detail → finish; each labelled with its stage name, radius, stroke count |
| **Palette** | the score's `P name r g b` pigment lines, ranked by USAGE (sum of mix weights over all strokes) | top 6 swatches; `palette: image` pigments get the nearest CLASSIC pigment name from a masstone table (titanium white, cadmium yellow, yellow ochre, burnt sienna, ultramarine, phthalo green, …) with the measured sRGB under it |
| **Brushwork concept** | the picture's contour drawing (the painter's own `draw_contours` / pen-ink path over the armature) + the stroke FLOW field + per-region mark stats | arrows along the dominant flow per region; a legend of the mark TYPES the run used (hair strands, dry-brush skips, wash sweeps, dabs, long sweeps) with their counts and where |

### 3.2 Choosing the four insets

Four regions that are both prominent and DIFFERENT, each a fact of the run:

1. **The focal plane** — the face mask's centroid (or the saliency peak when there is no face). Label from
   the measured surface: thin + smooth → "sfumato"; detail passes dominant → "fine modelling".
2. **The thickest paint** — the height field's maximum region OFF the face (the beard, the embroidery, a
   lit fold). Label: "impasto / ridges" (+ "plowed", "cast shadow" when `ridges` is on).
3. **The most textured passage** — the busy field's peak in the BACKGROUND (cobbles, bark, foliage).
   Label from the marks there: dry-brush skips → "scumble / dry brush"; dabs → "textured dabs".
4. **The thinnest / lightest passage** — the height field's minimum among the lights (a window, the sky).
   Label: "glaze over the ground" (oil) / "reserved paper" (watercolour); weave visible → "+ linen".

Non-maximum suppression keeps the four at least a quarter-sheet apart. Each inset's caption carries two
numbers from the run (e.g. "height 36% of peak · 4 px marks 61%"), so a reader can check it.

### 3.3 Rendering the sheet

The bookart suite already typesets through **Typst** (`title-page` / `cover` / `book`): the sheet is a
Typst template (`assets/outcome_sheet.typ`) fed with the crops, the pass stack, the swatches and the
contour drawing; compiled to PNG (and PDF on request). Lens crops are Typst circles with image clipping.
Fonts: the ones bookart ships. Medium-specific vocabulary (oil vs watercolour vs ink) comes from a small
per-medium label table.

## 4. CLI / plan

```
--analysis [PATH]      analysis: true | "path.md"
--outcome  [PATH]      outcome:  true | "path.png" | "path.pdf"
--outcome-insets N     outcome_insets: 4        (2..6)
--outcome-title TEXT   outcome_title: "Old Street Duet"   (default: the source file's stem)
```
`true` places the artefact beside the output image (`out.md`, `out_sheet.png`). Both default off. The
default paint path is untouched (the artefacts read the result; they never change it) — guard 0.

## 5. Phases

- **P0 — analysis.md** (`report.rs`) — BUILT: the six sections from existing data.
- **P0.5 — insights** (§2b): the LLM pass over the report with the versioned system prompt; hosted/Ollama
  via the existing enrichment layer; privacy line.
- **P1 — the sheet, data panels**: insets chosen by §3.2, palette ranking + classic-name table, pass stack;
  Typst template; PNG out.
- **P2 — the brushwork concept panel**: contour drawing + flow arrows + mark-type legend.
- **P3 — polish**: watercolour/ink vocabularies, optional LLM prose captions (hosted → the prose leaves the
  machine; Ollama keeps it local — say so in the docs), `plakat ui` hook.

## 6. Acceptance

- every number on the sheet is reproducible from the `.strokes` + the render (a test regenerates the
  analysis from a replay and diffs it);
- the four insets are distinct planes on three different pictures (night couple, market, portrait);
- default paint path byte-identical (guard 0); one heavy job at a time on the bench.

## 7. Non-goals

- Inventing palette names or techniques the run did not use; a "corrected" sheet that shows what the
  picture should look like; running a model to judge the picture.
