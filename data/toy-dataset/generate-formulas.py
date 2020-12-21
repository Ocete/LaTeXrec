from nltk import CFG, ChartParser
from nltk.parse.generate import generate
from random import choice
import sys


def produce(grammar, symbol, d=10):
    words = []
    productions = grammar.productions(lhs=symbol)
    production = choice(productions)
    for sym in production.rhs():
        if isinstance(sym, str):
            words.append(sym)
        elif d >= 0:
            words.extend(produce(grammar, sym, d-1))
    return words


grammar = CFG.fromstring('''
S -> EXP
EXP -> EXP OP EXP | SIGMA '_{'LETTER'='NUM'}' EXP | '\sqrt{'EXP'}' | LETTER | NUM
LETTER -> 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'x' | 'y'
NUM -> '0' | '1' | '2' | '3' 
OP -> '+' | '-'
SIGMA -> '\Sigma'
''')

parser = ChartParser(grammar)

N = int(sys.argv[1])
min_len = int(sys.argv[2])
max_len = int(sys.argv[3])

formulas = list()
gr = parser.grammar()
n = 0
while n <= N:
    expr = produce(gr, gr.start(), d=20)
    expr = ' '.join(expr)
    if len(expr) >= min_len and \
       len(expr) <= max_len and \
       expr not in formulas:
        formulas.append(expr)
        n += 1

        if n % 100 == 0:
            print('Generated ', n, file=sys.stderr)

for f in formulas:
    print(f)
