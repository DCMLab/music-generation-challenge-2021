import random

class Grammar:
    """Symbol -> Word (or list of Words)"""
    form = {
        '$': [['A', 'B', 'A'], ],
        'A': [['a', 'a'], ['a', 'a\''], ['a', '_']],
        'B': [['b', 'b'], ['b', 'b\''], ['b', '_']],
        'C': [['c', 'c'], ['c', 'c\''], ['c', '_']],
        'a': [['a_l', 'a_l\'','_','_'], ['a_l', 'a_l\'','_','_']],
        'a\'': [['a\'_l', 'a\'_m\'','_','_'], ['a\'_l', '_'],'_','_'],
        'b': [['b_m', 'b_n','_','_'], ['b_m', '_','_','_']],
        'b\'': [['b\'_m\'', '_'], ['b\'_m', 'b\'_m\'']],
        'c': [['c_n', 'c_n\'','_','_'], ['c_n', '_'],'_','_'],
        'c\'': [['c\'_n\'', '_','_','_'], ['c\'_n', 'c\'_l'],'_','_'],
        '_': [['_']],
    }

    Harmony = {
        '$': [['TD', 'TT']],
        'TD': [['|I', 'ii','V(HC)'],['|I', 'IV','V(HC)'],['|I','V(HC)']],
        'TT':[['|I','I(AC)'],['|I','V','I(AC)']],
        '|I':[['|I']],
        'I': [['V', 'I'], ['I']],
        'V': [['ii', 'V'], ['V']],
        'IV': [['IV']],
        'ii': [['vi', 'ii'], ['ii']],
        'vi': [['iii, vi'], ['vi']],
        'I(AC)': [['I(AC)']],
        'V(HC)': [['V(HC)']],
    }

    Contour = {
        '$': [['H']],
        'U': [['U', 'H'], ['H', 'U']],
        'H': [['U', 'D'], ['D', 'U']],
        'D': [['D', 'H'], ['H', 'D']],
        '_': [['_']],
    }

    Rhythm = {
        '$': [['opening','any_bar','any_bar','cadence']],
        'opening': [[['e', 'e'],['q'],['e','e']]],
        'any_bar': [[['e', 'e'],['e','e'],['e','e'],], [['s','s','s','s'],['e','e'],['s','s','s','s'],]],
        'cadence': [[['s','s','s','s'],'q','r_q'],[['e','s','s'],'q','r_q'],[['e','e'],'q','r_q']],
        'h':[['h']],
        'q': [['q']],
        'e': [['e']],
        'r_q':[['r_q']],
        #'edot':[['edot']],
        's':[['s']]
    }

    Key = {
        '$': [['D']]
    }

    GuideTone = {
        '$': [['1,3'], ['1,5'], ['3,5']],
    }

    @staticmethod
    def make_deterministic(dict):
        new_dict = {}
        for key, value in dict.items():
            new_dict[key] = random.choice(value)
        return new_dict