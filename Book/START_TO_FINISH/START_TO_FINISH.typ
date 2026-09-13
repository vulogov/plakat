// plakat — A Poster, Start to Finish : master file.
//
// Compile with:
//   typst compile Book/START_TO_FINISH/START_TO_FINISH.typ
//
// Output: START_TO_FINISH.pdf. A narrative companion to plakat's reference
// docs: where the --help output and RFCs are topical, this book follows ONE
// image — a promotional poster, "NIGHT MARKET" — from `plakat init` to a
// finished, upscaled, provenance-etched print, touching every feature in the
// order a real designer reaches for it. Learn by watching one poster get made.
//
// Teaches with monospace terminal `screen()` mockups, the way the app is
// actually used.

#import "design.typ": *

#book((
  include "chapters/preface-why-plakat.typ",
  include "chapters/00-following-one-poster.typ",

  part(number: "I", title: "Setting Up"),
  include "chapters/01-installing-and-first-image.typ",
  include "chapters/02-the-prose-project.typ",

  part(number: "II", title: "Authoring in Prose"),
  include "chapters/03-writing-the-scene.typ",
  include "chapters/04-will-it-render.typ",
  include "chapters/05-fixing-the-prose.typ",

  part(number: "III", title: "Compiling & Running"),
  include "chapters/06-compile-to-scenario.typ",
  include "chapters/07-running-and-models.typ",
  include "chapters/08-tuning-the-image.typ",

  part(number: "IV", title: "Composition & Control"),
  include "chapters/09-more-than-one-figure.typ",
  include "chapters/10-connected-objects-and-relations.typ",
  include "chapters/11-style-personas-loras.typ",

  part(number: "V", title: "Editing & Finishing"),
  include "chapters/12-editing-what-you-have.typ",
  include "chapters/13-making-it-look-real.typ",
  include "chapters/14-upscale-and-sharpen.typ",
  include "chapters/15-provenance-and-export.typ",

  part(number: "VI", title: "Reference"),
  include "chapters/A-prose-directive-reference.typ",
  include "chapters/B-model-aliases-and-families.typ",
  include "chapters/C-the-polish-loop.typ",

  include "chapters/99-about.typ",
  include "chapters/99b-about-the-author.typ",
))
