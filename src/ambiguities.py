def find_all_ocurrences(text, word):
    """
    Yields all the positions of the given word in the text.
    """

    i = text.find(word)
    while i != -1:
        yield i
        i = text.find(word, i+1)

def remove_ambiguities(formula):
    """
    Removes all ambiguities from the given formula
    """

    names = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'max', 'min']
    changes = [ ('\\operatorname {{ {} }}'.format(x),
                 '\\{}'.format(x)) for x in names] + \
              [ ('^ { \\prime }', '\\\''), ('\\dagger', '\\dag'),
                ('\\lbrace', '\\{'), ('\\rbrace', '\\}')  ]

    # Transform '\\cal X' expresions to '\\mathcal { X }'
    indexes = list(find_all_ocurrences(formula, '\\cal'))
    cal_changes = []
    for i in indexes:
        # Jump over '\\cal ' (including the space after)
        i = i + 5
        if i < len(formula):
            j = formula.find(' ', i)
            if j == -1: j = len(formula)
            X = formula[i:j]
            cal_changes.append( ('\\cal {}'.format(X),
                                 '\\mathcal {{ {} }}'.format(X)) )

    # Apply this changes sorted by size: greater before smaller
    cal_changes = sorted(cal_changes, key=lambda x: -len(x[0]))
    changes = changes + cal_changes

    for (_from, _to) in changes:
        formula = formula.replace(_from, _to)

    return formula


def test_remove_ambiguities():
    """
    Small test for remove_ambiguities
    """

    test_cases = [('\\operatorname { sin } ^ { \\prime } and my \\dagger',
                   '\\sin \\\' and my \\dag'),
                  ('Try \\cal this and \\cal this_other_one',
                   'Try \\mathcal { this } and \\mathcal { this_other_one }')]

    passed = True
    for t, result in test_cases:
        obtained = remove_ambiguities(t)
        if obtained != result:
            passed = False
            print('Error in: --{}--. Obtained: --{}--. Should be --{}--'.format(
                t, obtained, result))

    if passed:
        print('All test cases passed sucessfully')
