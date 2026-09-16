// The finished book: import each page's `title-page` and render them in order.
#import "00-frontispiece.typ": title-page as frontispiece
#import "01-title.typ": title-page as title-page-main
#import "02-chapter1.typ": title-page as chapter-one
#import "03-section1.typ": title-page as section-one
#import "04-section2.typ": title-page as section-two
#import "05-chapter2.typ": title-page as chapter-two

#frontispiece
#pagebreak()
#title-page-main
#pagebreak()
#chapter-one
#pagebreak()
#section-one
#pagebreak()
#section-two
#pagebreak()
#chapter-two
