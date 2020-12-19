import sys
import hashlib
from multiprocessing import Pool
import subprocess
from pathlib import Path

threads = 7
formulas_file = sys.argv[1]
im2latex_file = sys.argv[2]
output_dir = Path(sys.argv[3])

latex_template = r"""
\documentclass[preview]{standalone}
\usepackage{amsmath}
\begin{document}

\begin{displaymath}
%s
\end{displaymath}

\end{document}
"""

"""
pdflatex formula.tex -shell-escape
pdfcrop formula.pdf formula.pdf
convert -density 300 formula.pdf -flatten -quality 90 -quiet formula.png
"""


def formula_to_image(formula):
    latex = latex_template % formula
    name = hashlib.sha1(formula.encode('utf-8')).hexdigest()[:15]

    pngname = output_dir / (name+'.png')
    if pngname.exists():
        print('File {} exists, skipping'.format(pngname))
        return name+'.png'

    texname = output_dir / (name+'.tex')
    with open(texname, 'w') as texfile:
        texfile.write(latex)

    subprocess.call([
        'pdflatex',
        '-interaction=nonstopmode',
        '-halt-on-error',
        '-output-directory='+str(output_dir),
        str(texname)
    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)

    pdfname = output_dir / (name+'.pdf')
    subprocess.call([
        'pdfcrop',
        str(pdfname),
        str(pdfname)
    ])

    subprocess.call([
        'convert',
        '-density', '300', str(pdfname),
        '-flatten', '-quality', '90', '-quiet', str(pngname)
    ])

    return name+'.png'

if __name__ == '__main__':
    output_dir.mkdir(exist_ok=True, parents=True)
    
    formulas = open(formulas_file).read().split("\n")

    pool = Pool(threads)
    names = list(pool.imap(formula_to_image, formulas))

    im2latex_f = open(im2latex_file, 'w')
    for i, name in enumerate(names):
        im2latex_f.write(str(i)+' '+name+'\n')
        
