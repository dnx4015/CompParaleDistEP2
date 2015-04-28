# make -f Makefile_pdflatex

R = relatorio
B = bibliografia
C = capa

all: $(R).pdf clean

pdf: $(R).pdf

$(B).bbl: $(B).bib
	pdflatex $(R)
	bibtex $(R)
	pdflatex $(R)

$(R).pdf: *.tex $(B).bbl
	pdflatex $(R)

clean:
	rm -f $(R).log $(R).aux $(R).bbl $(R).blg $(R).toc $(R).out $(C).aux 

