#import "../design.typ": *
#chapter(number: 1, title: "Installing plakat, and Your First Image")

#dropcap("E")very project starts the same way: get the tool onto your machine,
confirm the machine can run it, and make *one* image so you know the whole chain
works before you invest in a scene. This chapter does exactly that. By the end you
will have plakat installed, a clean bill of health from `doctor`, and the first
frame of our poster — an empty lantern-lit lane — sitting in an `out/` folder.

#section("Getting plakat")

plakat ships three ways. Pick whichever suits how you like to install software;
they all land you at the same `plakat` command.

#subsection("From crates.io (the Rust way)")

If you have a Rust toolchain, one command builds and installs the current
release. It compiles from source, so budget a few minutes the first time. One
thing to get right here: `cargo install` builds a *CPU-only* binary unless you ask
for a GPU backend — so add `--features metal` on Apple Silicon (or `--features cuda`
on NVIDIA). See the callout below for why this is a build choice, not a runtime one.

#screen(caption: "Install from crates.io — with a GPU backend")[```
  $ cargo install plakat --locked --features metal    # Apple Silicon
  # NVIDIA:  cargo install plakat --locked --features cuda
  # plain `cargo install plakat` installs too — but renders on the CPU (slow)
      Updating crates.io index
     Compiling plakat v6.30.0
      Finished release [optimized] target(s)
     Installed /Users/you/.cargo/bin/plakat
```]

#subsection("A prebuilt binary (no toolchain)")

Every release also publishes prebuilt archives on GitHub — one per platform, each
with the right GPU backend already baked in (Metal on the Apple-Silicon archive,
CUDA on the `...-cuda` one). Grab the one for your machine, unpack it, and put the
binary on your `PATH` — nothing else to select.

#screen(caption: "Download a release binary")[```
  # macOS, Apple Silicon
  $ curl -LO https://github.com/vulogov/plakat/releases/download/\
         v6.30.0/plakat-v6.30.0-aarch64-apple-darwin.tar.gz
  $ tar xzf plakat-v6.30.0-aarch64-apple-darwin.tar.gz
  $ sudo mv plakat /usr/local/bin/
```]

The release carries five builds: Apple Silicon (`aarch64-apple-darwin`), Linux on
ARM and x86 (`aarch64`/`x86_64-unknown-linux-gnu`), a CUDA Linux build
(`...-cuda`), and Windows (`x86_64-pc-windows-msvc.zip`). A `SHA256SUMS` file sits
beside them if you want to verify the download.

#subsection("From source")

To hack on plakat, or to pick your own feature set, clone and build. The default
build includes the interactive UI, the photo manager, and the fractal engine — but
*no GPU backend*, so add one for real speed: `--features metal` on Apple Silicon,
`--features cuda` (or `cudnn`) on NVIDIA. Trim the defaults with
`--no-default-features` for a leaner CLI binary.

#screen(caption: "Build from source — with the Metal backend")[```
  $ git clone https://github.com/vulogov/plakat
  $ cd plakat
  $ cargo build --release --features metal   # Apple Silicon; --features cuda on NVIDIA
  $ ./target/release/plakat --version
  plakat 6.30.0
```]

Leave the feature off and you get a working binary that renders on the *CPU* —
correct, but minutes per image where the GPU is seconds. `plakat doctor` will report
`built cpu` so you can catch it at a glance.

#callout(label: "The backend is a build choice, not runtime magic")[
  plakat does *not* conjure a GPU backend from bare hardware. The backend is chosen at
  *build time*: a prebuilt release binary has the right one baked in for its platform,
  but `cargo install plakat` and a plain `cargo build --release` are *CPU-only* — no
  GPU backend is on by default. Add `--features metal` (Apple Silicon) or
  `--features cuda`/`cudnn` (NVIDIA) to get one. What plakat *does* auto-detect is which
  of the *compiled-in* backends to use at runtime; `--device` overrides that per
  command. Skip the feature on a Metal Mac and you get a slow CPU binary — and `doctor`'s
  "build vs runtime" line (below) is exactly what tells you so.
]

#section("Checking the machine: `doctor`")

Before downloading a multi-gigabyte model, ask plakat whether this machine is set
up correctly. `doctor` is fully offline by default — it inspects your hardware,
your build-versus-runtime device alignment, cache disk usage, and whether the
optional tools it can use (ffmpeg, API tokens) are present. It never prints a
token's value, only whether one is set.

#screen(caption: "plakat doctor")[```
  $ plakat doctor
   INFO Using Metal device

    text-to-image hardware
      backend   metal · features metal
      memory    24 GB unified · budget ~18 GB
      status    healthy — SD 1.5 / SDXL / SD3.5 (low-mem) run here

    build vs runtime
      built     metal
      runtime   metal · aligned

    cache
      HF cache  ~/.cache/huggingface · 0 B (empty)

    tokens
      HF_TOKEN        not set (only gated models need it)
      CIVITAI_TOKEN   not set
```]

The *build vs runtime* block is the one to read first on a machine you built yourself.
`built metal · runtime metal · aligned` means the Metal backend is compiled in and in
use. If instead you see `built cpu` on an Apple-Silicon Mac, you installed without
`--features metal` — the binary works but renders on the CPU. Rebuild with the feature
and this line flips to `metal · aligned`. It is the definitive answer to "why is my
Mac rendering so slowly?"

Two follow-up questions are worth asking on a new machine. *Which models can this
hardware actually run?* and *how fast will a render be?* plakat answers both
without downloading anything.

#screen(caption: "Will it run, and how fast?")[```
  $ plakat doctor --capability
    text-to-image hardware
      backend   metal · budget ~18 GB
      sd15          runs      · 512² native
      sdxl          runs      · 1024² native
      sd35-medium   runs      · needs HF_TOKEN
      sd35-large    tight     · needs ≥32 GB; use sd35-medium
      flux-dev      won't fit  · → flux-dev-gguf Q4 + --quantize-t5

  $ plakat doctor --benchmark
    conv2d / matmul / resize at SD-typical shapes …
      extrapolated SD 1.5 wall-time  ~9 s / image (28 steps)
```]

#term("Capability report")[
  `plakat doctor --capability` probes your RAM and backend, derives each model's
  weight size, and judges *runs / tight / won't-fit* — naming the one lever that
  helps when a model is tight. It is the fastest way to know what you can make on
  this machine before you commit to a download. Add `--json` for a scriptable
  version.
]

#section("Choosing the first model")

Our first image needs a model. plakat recognises short *aliases* — `sd15`,
`sdxl`, `flux-schnell`, and many more — each mapping to a full HuggingFace repo.
The safest starting model is Stable Diffusion 1.5: it is small, open (no token),
fast, and runs on almost anything. We will graduate to SDXL for the finished
poster later; for a first proof of life, `sd15` is perfect.

#screen(caption: "See every model alias plakat knows")[```
  $ plakat models aliases
  ── SD 1.5 ──
    sd15, sd-1.5            → stable-diffusion-v1-5/... (base)
  ── SDXL ──
    sdxl                    → stabilityai/stable-diffusion-xl-base-1.0
    sdxl-turbo             → adversarial-distilled, 1–4 steps, no CFG
    pony                   → Pony Diffusion v6 XL (SDXL fine-tune)
  ── Flux ──
    flux-schnell           → 4-step, Apache-2.0
    flux-dev               → gated; larger, higher quality
```]

You do not download a model by hand — plakat pulls the weight files it needs on
first use and caches them under your HuggingFace cache. If you would rather fetch
them ahead of time (say, before going offline), `plakat models pull` does it, and
`plakat models ls` shows what is already cached.

#section("The first image")

Now the moment of truth. We ask for the seed of our poster — the empty lane, no
people yet — with a single, plain sentence. Do not overthink this prompt; we will
rebuild it properly, in prose, over the next chapters. Right now we only want to
see the machine produce a picture.

#screen(caption: "Generate the empty lane")[```
  $ plakat generate "a lantern-lit cobbled market lane at night, \
      soft fog, warm amber glow, no people" \
      --model sd15 --size 512x512 --seed 1000
   INFO Using Metal device
   INFO downloading stable-diffusion-v1-5 … 4 files, 1.9 GB
   INFO 28 steps ████████████████████ 28/28  8.7s
   ✓ wrote ./out/plakat-1000.png
```]

#figure_img("assets/01-empty-lane.png", "Image 01 — the first render: the empty lantern-lit lane. Rough and generic on a one-line prompt, but unmistakably the right scene.")

The first run pays a one-time download for the model; later runs skip straight to
generation. When it finishes, open `./out/plakat-1000.png`. You should see a
narrow cobbled lane, a lantern or two, fog softening the far end — rough, a little
generic, but unmistakably the right scene. That roughness is expected: this is a
one-line prompt on the smallest model. The whole rest of the book is about turning
this proof of life into a poster.

A word on what just happened, because it shapes everything after. `plakat generate` is
the *scratchpad*: one sentence in, one throwaway image out, nothing kept. It is perfect
for a first pulse-check like this one, and you will keep it nearby for quick scouting.
But it is *not* how we will make the poster. From the next chapter on, the images we
keep come out of a small, repeatable loop — write prose, compile it, run the scenario —
and `generate` steps aside to be what it is best at: a place to try something fast
before committing it to the source.

#subsection("What plakat wrote beside the image")

Look in `out/` and you will find more than a PNG.

#screen(caption: "What a generation leaves behind")[```
  $ ls out/
  plakat-1000.png      the image
  plakat-1000.json     the recipe: model, seed, steps, sampler, size
```]

Every image carries its own recipe. The `.json` sidecar — and a matching block of
metadata written *inside* the PNG — records exactly how the image was made, so you
can reproduce it, tweak one setting, or hand it to someone else. Two commands live
on this metadata: `plakat metadata out/plakat-1000.png` reads it back, and `plakat
clone out/plakat-1000.png` prints the exact command to regenerate it. We will lean
on both in the finishing chapters.

#callout(label: "Seeds make it repeatable")[
  The `--seed 1000` is why your image will match the recipe next time. Leave the
  seed off and plakat picks a random one (and records it, so you can recover it).
  Fix the seed whenever you want to change *one* thing and see only that change —
  which, in image work, is most of the time.
]

#section("A note on speed and patience")

On Apple Silicon or a decent NVIDIA card, an SD 1.5 image is a handful of seconds.
On CPU it can be minutes. plakat will finish either way, but if you are on CPU,
prefer `sd15` at `512x512` for everything in the early chapters, keep `--count 1`
while you experiment, and save the bigger models for when the composition is
settled. The single most expensive mistake in image work is rendering the wrong
thing slowly — which is precisely the mistake the next four chapters are built to
prevent.

#recap((
  [Install plakat three ways — `cargo install plakat`, a prebuilt release binary,
  or from source. The GPU backend is a *build* choice: release binaries bake in the
  right one per platform, while `cargo install`/from-source are *CPU-only* unless you
  add `--features metal` (Apple Silicon) or `--features cuda` (NVIDIA). At runtime
  plakat auto-selects among the backends compiled in; `doctor` shows `built …` so you
  can confirm you got the fast one.],
  [`plakat doctor` health-checks the machine offline; `--capability` says which
  models *run / are tight / won't fit* here, and `--benchmark` estimates render
  time — all before any download.],
  [Models are named by short *aliases* (`sd15`, `sdxl`, `flux-schnell`, …) and
  pulled on first use; `sd15` is the safe, fast, token-free starting model.],
  [`plakat generate "<sentence>" --model sd15 --seed 1000` makes the first image
  and writes a `.png` plus a `.json` recipe you can later read (`metadata`) or
  reproduce (`clone`).],
  [Fix a seed when you want to change one thing at a time — and never render a big
  version of something you haven't proven small first.],
))
