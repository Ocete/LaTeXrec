all: propuesta.pdf bitácora.pdf

propuesta.pdf: propuesta.md bibliography.bib
	pandoc propuesta.md \
	--template pandoc-latex-template/eisvogel.tex \
	--citeproc \
	--bibliography bibliography.bib \
	-o propuesta.pdf

bitácora.pdf: bitácora.md bibliography.bib
	pandoc bitácora.md \
	--template pandoc-latex-template/eisvogel.tex \
	--citeproc \
	--bibliography bibliography.bib \
	-o bitácora.pdf
