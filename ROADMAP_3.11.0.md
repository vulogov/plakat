# plakat 3.11.0 — roadmap (in progress)

**Theme — the Ken Burns slideshow.** 3.10 gave the slideshow shuffle + rating-weighted dwell. 3.11
adds the documentary **Ken Burns** effect: each slide slowly pans and zooms across the frame, turning
a still review into gentle motion.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — Ken Burns pan/zoom

- [x] **Ken Burns effect** (`k` in the slideshow) — each slide animates a smooth crop from one framing
      to another (random pan + zoom, smoothstep-eased), rendered by cropping the cached `working_base`
      each event-loop tick (`tick_ken_burns` → `render_ken_burns`; no re-decode). The motion spans the
      slide's (rating-weighted) dwell; each new slide picks a fresh `kb_from`/`kb_to` framing (window
      kept inside the image). A `🎥` marker in the `▶` badge + status; off by default (opt-in motion),
      cleared when the slideshow stops or you leave the image view. Pure crop-rect / easing math in
      `kb_lerp` + `kb_crop_rect`; 1 test.

## Ground rules (unchanged)

- Non-destructive; the effect is view-only (never alters the file or the edit stack).
- Cheap: reuse the per-image `working_base` cache; bound the render resolution like normal viewing.
- Verify-safe: default CLI image output byte-identical; new logic lands with unit tests where it's
  pure (the crop-rect / easing math).
