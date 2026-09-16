// The finished book: import each page's `title-page` and render them in order.
#import "00-frontispiece.typ": title-page as frontispiece
#import "01-title.typ": title-page as title-page-main
#import "02-book.typ": title-page as book-title
#import "03-chapter1.typ": title-page as chapter-one
#import "04-section1.typ": title-page as section-one
#import "05-section2.typ": title-page as section-two
#import "06-chapter2.typ": title-page as chapter-two

#frontispiece
#pagebreak()
#title-page-main
#pagebreak()
#book-title
#pagebreak()
#chapter-one
#pagebreak()
#section-one
#pagebreak()
#section-two
#pagebreak()
#chapter-two
