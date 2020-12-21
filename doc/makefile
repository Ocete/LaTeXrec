all: propuesta.pdf

propuesta.pdf: propuesta.md bibliography.bib
	pandoc propuesta.md \
	--template pandoc-latex-template/eisvogel.tex \
	--citeproc \
	--bibliography bibliography.bib \
	-o propuesta.pdf
