#import "../design.typ": *
#appendix(letter: "A", title: "The Prose Directive Reference")

This is the working reference for the prose you write in a `prompts.txt` — every
directive the book used, grouped by what it controls, plus the in-prose syntax for
weights, scheduling, and structure. Directives are `key: value` lines; free text
between them is the scene's prose. Global-block directives set project defaults;
most can be overridden per scene.

#section("Project & output")

#chord_table((
  chord_row("model:", "Model alias or HF repo (sd15, sdxl, flux-schnell, …). Global; a scene may override for itself."),
  chord_row("size:", "Render size, e.g. 1024x1024. Prefer the family's native square."),
  chord_row("seed:", "Base seed. Per-image seeds step from it; fix it for reproducibility."),
  chord_row("count:", "Images per task."),
  chord_row("out:", "Output directory (created on first run)."),
  chord_row("enhancer:", "Enhancement provider: local · deepseek · gemini · auto."),
  chord_row("device:", "auto · metal · cuda · cpu."),
))

#section("Sampler & quality")

#chord_table((
  chord_row("steps:", "Denoising steps (SDXL: 28–35 is plenty)."),
  chord_row("guidance:", "Classifier-free guidance (~7 a good middle; 0 for turbo)."),
  chord_row("scheduler:", "default · ddim · euler-a · unipc."),
  chord_row("clip-skip:", "1 (default) or 2 (SD 1.5/2.1 community checkpoints)."),
  chord_row("negative:", "Scene-specific terms to avoid, added to the automatic quality set."),
))

#section("Reuse & structure")

#chord_table((
  chord_row("name:", "The scene's name (becomes the task name and output stem)."),
  chord_row("skip:", "true to skip a scene without deleting it."),
  chord_row("component.<n>:", "A named, reusable scene fragment. Reference it from a DIRECTIVE (below), not by bare mention. Global or scene-local (local overrides)."),
  chord_row("composition:", "Comma-list of components joined in order, BEFORE the free-text prose (compose-then-prose)."),
  chord_row("scene.<n>: / weather.<n>:", "Named variant axes; each render picks one of each."),
  chord_row("loras:", "LoRA stack, name:strength (e.g. ink-poster-xl:0.65)."),
  chord_row("lora-trigger:", "Trigger word(s) a named LoRA needs to activate."),
  chord_row("@include path", "Inline another prose file (with optional key=value passing)."),
))

#section("Composition & control")

#chord_table((
  chord_row("foreground:", "Comma list of intended focal figures; fixes the figure count."),
  chord_row("objects:", "Comma list of props that must render SEPARATELY from figures (the cart)."),
  chord_row("relate:", "A relationship between two components: <a> <verb> <b>. Placement verbs (behind, beside, facing) stage layout; ATTACH verbs (holding, wearing, carrying) fold b INTO a's description so a held prop doesn't float."),
  chord_row("control-generate:", "Enable the composition pass; value names the draft model (sdxl/sd15)."),
  chord_row("control-generate-size:", "Draft resolution (match the draft model: 1024² for sdxl, 512² for sd15)."),
  chord_row("control-generate-regional:", "true (default) paints each figure in its own area; false blends more."),
  chord_row("control-generate-figure-poses:", "One posture per foreground figure (standing, leaning, sitting…)."),
  chord_row("control-generate-figure-contacts:", "Limb-level touches between figures (arm-around, holding-hands…)."),
))

#section("Finishing")

#chord_table((
  chord_row("naturalize:", "Scene-tuned finishing pass baked into compile (grade, grain, vignette, repaint-protect=figures)."),
))

#section("In-prose syntax")

These live *inside* the free-text prose, not as `key:` lines.

#chord_table((
  chord_row("(phrase:1.3)", "Attention weight — above 1.0 emphasises, below 1.0 eases back. Preserved through compile."),
  chord_row("BREAK", "Encode what precedes and follows independently, so distinct elements don't bleed."),
  chord_row("[a:b:when]", "Prompt scheduling — swap a for b once the render passes the fraction `when`."),
  chord_row("[a|b]", "Alternate between a and b each step."),
))

#section("The commands that act on prose")

#chord_table((
  chord_row("compile FILE", "Prose → scenario. --out, --explain, --dry-run, --check, --diff, --watch."),
  chord_row("compile --analyze", "Feasibility report (grade + risks + fixes); renders nothing."),
  chord_row("compile --analyze --fix", "Apply safe fixes, back up to FILE.N, record the smysl corpus."),
  chord_row("compile --trace \"…\"", "Explain why a phrase is in — or gone from — the prompt (no render)."),
  chord_row("compile --smysl", "Also write the provenance sidecar beside the scenario."),
  chord_row("compile --make-composition", "Suggest a generation strategy + a cleaner prose rewrite."),
  chord_row("scenario FILE", "Render every task in a compiled scenario."),
))

#callout(label: "The one habit")[
  Edit the prose; treat the scenario as compiled output. Every directive here lives
  in a file you write and read, so `--analyze`, `--fix`, and `--trace` can reason
  about your intent — which is the whole reason the prose project exists.
]
