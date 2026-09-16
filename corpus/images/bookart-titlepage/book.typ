// The finished book: import each page's `title-page` and render them in order.
#import "title.typ": title-page
#import "chapter.typ": title-page as chapter-page

#title-page
#pagebreak()
#chapter-page
