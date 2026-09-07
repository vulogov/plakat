# Rank — score and curate by aesthetic quality

`plakat rank` scores images by **aesthetic quality** and prints them
best-first, so you can sift a folder of generations down to the keepers
without eyeballing every file. The same scorer powers
`plakat generate --keep-best`, which prunes a batch automatically.

Scoring uses the **LAION aesthetic predictor** — a small MLP head on top
of CLIP ViT-L/14 embeddings, trained on human aesthetic ratings. Scores
land roughly in the **3–7** range; higher = more aesthetically pleasing.
It's a rough proxy, not a taste oracle, but it's a reliable first pass
for "which of these 20 renders are worth a second look."

## Ranking a folder

```bash
# Score everything in ./out, print best-first
plakat rank ./out

# Only show the top 5
plakat rank ./out --top 5

# Machine-readable output for scripting
plakat rank ./out --json
```

`plakat rank` accepts any mix of **files and directories**. Directories
are scanned **non-recursively** for `.png` / `.jpg` / `.jpeg` / `.webp`.
You can also pass files directly:

```bash
plakat rank a.png b.png c.png --top 2
```

The first run downloads the CLIP model + the aesthetic predictor head;
subsequent runs reuse the cache.

| Flag | Meaning |
|---|---|
| `--top <N>` | Print only the best N (default: all, best-first) |
| `--json` | Emit structured JSON (path + score) instead of the table |

## Auto-curating a batch

`plakat generate` can score a batch and keep only the best, in one
command. Generate N images, keep the aesthetically best K, delete the
rest:

```bash
# Generate 8, keep the best 2
plakat generate "a scenic mountain lake" --count 8 --keep-best 2
```

The pruned files include their `.json` metadata sidecars, so you're not
left with orphaned recipes. `--keep-best` only ever deletes files **this
run created** — any pre-existing files in the output directory are never
touched.

A quick worked flow:

```bash
# Rank a folder, show the top 5
plakat rank ./out --top 5

# Generate 8, keep the best 2
plakat generate "a scenic mountain lake" --count 8 --keep-best 2
```

The aesthetic score is the **first sort key** for the forthcoming
collection manager, so ranking today lines your library up for
quality-first browsing later.

## Vision judge & coach — the corrective aide

Aesthetic and AI-tell scores reward a smooth, pleasing look — which is
*anti-correlated* with the things that actually break: bad anatomy,
hallucinated extras, wrong attributes, ignored negatives. A smooth image
with a missing arm outscores a slightly rougher, anatomically-correct one.
`--vision` fixes that by asking a **vision LLM** to judge what a pixel
statistic can't:

```bash
# Score 0–10 on QUALITY + FAITHFULNESS to the prompt (anatomy, attributes,
# counts, no hallucinations) — art style ignored. Higher = better.
plakat rank ./out --vision \
  --prompt "a man and a woman sitting on a park bench, autumn" \
  --top 5
```

Needs a vision provider: `--provider auto` (the default) uses **Gemini**
when `GEMINI_API_KEY` is set; `--provider deepseek` uses any
OpenAI-compatible vision endpoint in that slot. One vision call per image.

`--coach` turns ranking into an *aide*: for each image it names the
concrete defects and proposes a corrective **prompt + negative** rewrite —
keeping your subject, scene, and style. Point it at the frame you're
unhappy with:

```bash
plakat rank ./out/broken_frame.png --coach \
  --prompt   "a man and a woman sitting on a park bench, autumn" \
  --negative "missing limbs, extra fingers, deformed hands"
# ★  4.20  broken_frame.png
#         ↳ defects  woman's left arm missing below elbow; third hand on bench
#         ↳ prompt   ... , both arms fully visible, natural relaxed pose, correct anatomy, exactly two people
#         ↳ negative missing limbs, extra fingers, deformed hands, third arm, floating hand
```

Combine `--vision --coach` to both score and critique. The same judge and
coach drive `ranking: by=vision coach=on` inside a scenario (see the
[Scenarios tutorial](SCENARIOS_TUTORIAL.md)) — there they also *regenerate*
until enough frames pass, feeding the coach's rewrite into the next rounds.

## Where to next

- **Generate the batches you'll rank** → [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md)
- **Batch production from HJSON** → [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md)
