# Looks catalog (v0.25)

Each entry in `catalog.json` describes an **art medium** preset.
Looks are *prescriptive* (you pick one); contrast with `--style`
which is *detective* (CLIP-H matches a reference photo).

Loaded at startup. User-added entries from
`$CONFIG_DIR/looks/*.json` shadow bundled entries by `name`.

## Field shape (`LookSpec`)

```jsonc
{
  "name":            "watercolor",        // kebab-case identifier; --look NAME
  "display_name":    "Watercolor",        // shown in --list output and logs
  "description":     "Transparent ...",   // one-line summary
  "prompt_prefix":   "watercolor ...",    // prepended to user prompt
  "prompt_suffix":   ", on cold-...",     // appended to user prompt
  "negative_extras": "photographic, ...", // appended to --negative
  "scheduler_hint":  "dpmpp-2m",          // sampler suggestion
  "steps":           32,                  // step-count suggestion
  "guidance":        6.0,                 // CFG-scale suggestion
  "lora_query": {                         // drives auto-LoRA discovery
    "tags":     ["watercolor"],           // exact-match Civitai tags
    "keywords": ["watercolor", "wash"]    // fuzzy-search free text
  },
  "base_compat": null                     // null = all bases; or list of
                                          // "sd15"/"sdxl"/"flux"/"sd3"
}
```

## Override semantics

Every scalar field is **optional**. The preset only fills fields
the user hasn't already populated on the command line / scenario
/ bund script. Matches the v0.14 `--fast` distillation-preset
rule: presets are suggestions, not overrides.

## LoRA discovery

`lora_query` fires only when `ctx.loras` is empty at generate
time. Source order: Civitai → HuggingFace Hub → local cache.
`--offline` short-circuits to cache + local scan.

## Adding your own look

Create `$CONFIG_DIR/looks/my-look.json` (single object, same
shape minus the outer `looks:` array — one file per look). The
loader picks it up at startup. Name conflicts shadow bundled
entries.

See `Documentation/RFC_v0.25_LOOKS_AND_GENRES.md` for the full
design.
