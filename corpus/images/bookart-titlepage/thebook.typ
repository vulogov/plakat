// ─────────────────────────────────────────────────────────────────────
// plakat bookart — a whole typeset BOOK (title page · chapters · folios · colophon).
// Compile:  typst compile <this-file>.typ
// ─────────────────────────────────────────────────────────────────────

#set page(
  width: 148mm, height: 210mm,
  margin: (inside: 22mm, outside: 18mm, top: 20mm, bottom: 22mm),
  footer: context align(center, text(size: 9.5pt)[#counter(page).display()]),
  header: context { if counter(page).get().first() > 1 { align(center, text(size: 8.5pt, tracking: 0.08em)[#smallcaps("The Open Sea")]) } },
)
#set text(size: 11pt)
#set par(leading: 0.72em, justify: true, first-line-indent: 1.4em)

#include "01-title.typ"
#counter(page).update(1)

#par[This little book is dedicated to all who have watched a coastline sink beneath the stern and felt the first long swell of the open sea.]

#pagebreak(weak: true)
#{
  set align(center)
  v(6%)
  image("rosette-1_crop.png", width: 26%)
  v(0.6em)
  text(size: 11pt, tracking: 0.14em)[#smallcaps("Chapter I")]
  v(0.5em)
  text(size: 19pt, weight: "bold")[#upper("The Departure")]
  v(1.6em)
}
#set par(first-line-indent: 0pt)
#par[#text(size: 2.6em, weight: "bold")[T]#h(0.04em)#smallcaps[he wind rose over] Plymouth Sound that morning, and the Amaranth swung to her anchor as though impatient to be gone. We had provisioned for a twelvemonth, and the hold was heavy with salt beef, biscuit, and casks of sweet water drawn from the Tavy.]

#set par(first-line-indent: 1.4em)
#par[At the turn of the tide we made sail. The town fell away, grey and small, and the Eddystone light stood up alone against the paling sky. The master set a course south by west, and the great business of the voyage was begun.]

#par[For three days we ran before a #emph[soldier's wind], and the people found their sea-legs and their spirits. On the fourth the glass began to fall.]

#v(1.2em)
#align(center, text(size: 11pt, tracking: 0.12em)[#smallcaps("The Falling Glass")])
#v(0.6em)
#set par(first-line-indent: 0pt)
#par[I noted the change in my journal that evening, in the cramped hand of a man #strong[braced against the roll]:]

#v(0.5em)
#pad(left: 2.5em, right: 2.5em)[#{
  set text(size: 10.5pt, style: "italic")
  set par(first-line-indent: 0pt)
  [Wind backing to the south-east, and a long swell from the same quarter. The mercury has dropped a full half-inch since the forenoon watch. I like it not.]
}]
#v(0.5em)
#par[By the first dog-watch the sky to windward had gone the colour of a bruise.]

#v(0.7em)
#align(center, text(size: 11pt, tracking: 0.5em)[\*\*\*])
#v(0.7em)
#par[We shortened sail while there was yet light to see the work, and I confess I have never been gladder of a well-drilled crew.]

#v(1.2em)
#align(center, image("dinkus_crop.png", width: 16%))

#pagebreak(weak: true)
#{
  set align(center)
  v(6%)
  image("rosette-1_crop.png", width: 26%)
  v(0.6em)
  text(size: 11pt, tracking: 0.14em)[#smallcaps("Chapter II")]
  v(0.5em)
  text(size: 19pt, weight: "bold")[#upper("The Storm")]
  v(1.6em)
}
#set par(first-line-indent: 0pt)
#par[#text(size: 2.6em, weight: "bold")[B]#h(0.04em)#smallcaps[y nightfall the sea] ran mountains high, and the Amaranth laboured in the troughs like a spent horse. We struck the topgallant masts and lay to under a close-reefed maintopsail, and still the water came aboard in green walls that swept the waist from rail to rail.]

#set par(first-line-indent: 1.4em)
#par[The mainmast sprung in the middle watch. I have never heard a sound so like a cannon, nor felt a deck so lifeless under my feet as in the moment the spar gave way. We cut the wreckage clear and rigged a jury-mast at dawn, our hands raw and our hearts low.]

#par[When the gale blew itself out we were four hundred miles from any reckoning, and glad only to be alive.]

#v(1.2em)
#align(center, image("dinkus_crop.png", width: 16%))

#pagebreak(weak: true)
#align(center + horizon, text(size: 9.5pt, tracking: 0.06em)[#smallcaps("Set in Libertinus · Printed at the Admiralty Press · MDCCXLI")])
