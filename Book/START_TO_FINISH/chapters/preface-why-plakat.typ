#import "../design.typ": *

#v(1cm)
#align(center)[
  #text(font: body_family, size: 20pt, weight: "bold", fill: ink_black, "Why plakat?")
]
#v(6mm)

Most image tools ask you to do the same thing: type a wish into a box, wait, and
hope. When the picture comes back wrong — a fused figure, a garbled hand, a scene
that was impossible from the first word — you change a word and hope again. The
compute is spent, the reasoning is gone, and nothing you learned survives to the next
attempt. It is a slot machine with a progress bar.

plakat is built on a different bet: that making a good image is *work you can reason
about*, not luck you wait on. It is a local, pure-Rust studio — every image is rendered
on your own machine, it phones nothing home, and it asks for no subscription. Its
optional steps that *reason about* your prose — enhancing a prompt, judging a render,
improving a scene — run on a small on-device model by default, so nothing need ever
leave your computer. The one time anything does is if *you* choose to route those steps
through a hosted language model; even then you can stay fully local with a model run
under Ollama. That choice is always yours, and Chapter 6 spells out exactly when text
leaves the machine and how to keep it from doing so. But its real idea is smaller and
more useful than any feature list. You
write your scene in *prose*, in a plain file you own. Before you spend a single render,
plakat reads that prose and tells you what will probably fail — and why. You fix the
prose, cheaply, and it remembers the reason for every change. Only when the scene is
sound do you commit the compute. The slot machine becomes a workbench.

That is the whole reason I built it. Not to have the most models or the loudest
demos — it has plenty of both — but to respect the two things a maker cannot get
back: the *time* a wrong render burns, and the *thread* of why a picture
turned out the way it did. A tool that tells you the scene is over-stuffed before you
wait three minutes to discover it, and that can answer "why is this phrase in my
prompt?" a month later, is a tool that treats your attention as worth something.

This book is the argument made concrete. Rather than catalogue what every command
does, it follows one image — a small night-market poster — from an empty project to a
signed, printable file, reaching for each tool exactly when the work asks for it. By
the end you will not merely know the flags; you will know the *order*, which is the
part no reference page can teach. If you have ever felt that image models make you
their operator instead of the other way around, this is the book — and plakat is the
tool — that hands the controls back.

#v(4mm)
#line(length: 100%, stroke: 0.5pt + ink_rule)
