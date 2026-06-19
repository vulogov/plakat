# `plakat compile` — Tera template pre-pass (COMPILE-2)

An optional [Tera](https://keats.github.io/tera/) template pass that runs *before*
the [`prompts.txt` parser](COMPILE.md). A `.tera` / `.j2` / `.jinja` input (or
`--template`) is rendered to a `prompts.txt` string, which then flows through the
normal compile pipeline unchanged. Generate a 50-character portrait series from one
data file, branch on fields, share macros — then compile + render as usual.

**Feature-gated.** Build with the `templates` feature; without it, a template input
errors clearly:

```bash
cargo build --release --features "metal templates"   # or: cargo install plakat --features templates
```

## Run

```bash
plakat compile series.tera --vars characters.json            # → series.hjson
plakat compile series.tera --vars characters.json --dump-rendered-only   # inspect the prompts.txt
plakat compile series.tera --var model=flux-dev --vars base.toml | plakat scenario -
```

## Context (later wins on key conflict)

1. **built-ins** — `{{ plakat.version }}`, `{{ plakat.input_stem }}`, `{{ plakat.input_path }}`
2. **`--vars <file>`** — JSON or TOML object; repeatable, later files win
3. **`--vars-env <PREFIX>`** — env vars with the prefix (stripped, lowercased): `PLAKAT_MODEL` → `{{ model }}`
4. **`--var KEY=VALUE`** — highest precedence; repeatable

## Filters & functions

| Filter | Does |
|---|---|
| `name \| scene_name` | slugify to a scene id (`"Lady Mireth"` → `lady_mireth`) |
| `list \| prompt_join` | join an array with `, ` |
| `s \| prompt_clean` | collapse stray commas/spaces |
| `n \| zero_pad(n=2)` | zero-pad an integer (`3` → `03`) |
| `s \| sentence_case` | capitalize the first letter |

| Function | Does |
|---|---|
| `scene_separator(title="…")` | emit a `# ── title ──…` comment line |
| `model_family(name="flux-dev")` | `"sd15"` / `"sdxl"` / `"flux"` / `"unknown"` |
| `include_raw(path="shared.txt")` | inline a file **verbatim** (not re-rendered) |

`{% include %}` / `{% import %}` resolve sibling template files in the input's
directory (for shared macro libraries).

## ⚠️ The blank-line pitfall (read this)

`prompts.txt` blocks are separated by **blank lines**. Tera's `{% for %}` and
`{% if %}` tags emit a newline by default — a spurious blank line **splits a block
in two**. Use the whitespace-trim markers `{%-` / `-%}`:

**Wrong** (the `{% for %}` newline splits every block):
```jinja
{% for c in characters %}
name: {{ c.name | scene_name }}

{{ c.appearance }}.
{% endfor %}
```

**Right** (`{%-` trims the newline before the tag; the block-separating blank line
is produced *explicitly* inside the loop body):
```jinja
{%- for c in characters %}

name: {{ c.name | scene_name }}
{{ c.appearance }}.
{%- endfor %}
```

When in doubt, `--dump-rendered-only` and eyeball the `prompts.txt` before spending
LLM calls.

## Example

`series.tera` + `series.json` in [`corpus/compile/`](../corpus/compile/) render a
two-character portrait series (mage + ranger, branched headers). `corpus/compile.sh`
proves it end to end: the Tera render is byte-stable vs the committed
`series.rendered.txt`, and `series.tera` (+ `series.json`) compiles to the committed
**`series.hjson`** scenario (2 character tasks), which `scenario --dry-run` validates.

Full compile reference: [`COMPILE.md`](COMPILE.md).
