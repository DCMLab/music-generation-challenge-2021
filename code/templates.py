import random


class Grammar:
    """Symbol -> Word (or list of Words)"""
    form = {
        # top
        '$': [['A', 'B']],
        # section
        'A': [['presentation', 'continuation']],
        'B': [['antecedent', 'consequent']],
        # phrase
        'presentation': [['basic_idea', 'basic_idea\'']],
        'continuation': [['cont', 'cadential']],
        'antecedent': [['basic_idea', 'contrasting_idea']],
        'consequent': [['basic_idea\'', 'contrasting_idea\'']],
        # sub-phrase
        'basic_idea': [['basic_idea']],
        'basic_idea\'': [['basic_idea_response'], ['basic_idea_repeat'], ['contrasting_idea']],
        'contrasting_idea': [['contrasting_idea']],
        'contrasting_idea\'': [['contrasting_idea\'']],
        'cont': [['fragment', 'fragment\''], ],
        'cadential': [['cadential_progression', 'cadence'], ],
        'basic_idea_repeat': [['basic_idea_repeat']],
        'basic_idea_response': [['basic_idea_response']],

        # 'A': [['a', 'a'], ['a', 'a\''], ['a', '_']],
        # 'B': [['b', 'b'], ['b', 'b\''], ['b', '_']],
        # 'C': [['c', 'c'], ['c', 'c\''], ['c', '_']],
        # 'a': [['a_l', 'a_l\'','_','_'], ['a_l', 'a_l\'','_','_']],
        # 'a\'': [['a\'_l', 'a\'_m\'','_','_'], ['a\'_l', '_'],'_','_'],
        # 'b': [['b_m', 'b_n','_','_'], ['b_m', '_','_','_']],
        # 'b\'': [['b\'_m\'', '_'], ['b\'_m', 'b\'_m\'']],
        # 'c': [['c_n', 'c_n\'','_','_'], ['c_n', '_'],'_','_'],
        # 'c\'': [['c\'_n\'', '_','_','_'], ['c\'_n', 'c\'_l'],'_','_'],
        # '_': [['_']],
    }

    Harmony = {
        # top
        '$': [['A', 'B'], ],
        # section
        'A': [['presentation', 'continuation']],
        'B': [['antecedent', 'consequent']],
        # phrase
        'presentation': [[['|I'], ['V']], [['|I'], ['I']]],
        'continuation': [[['V'], [['V'],['V','I(AC)']]]],
        'antecedent': [[['|I'],['V(HC)']], [['|I'],['V']]],
        'consequent': [[['|I'], ['I(AC)']]],

        # sub_phrase

        # sub-phrase ~ beat
        '|I': [['|I']],
        'I': [['V', 'I'], ['I']],
        'V': [['ii', 'V'], ['V']],
        '|IV': [['|IV']],
        'IV': [['IV']],
        'ii': [['vi', 'ii'], ['ii']],
        'vi': [['iii', 'vi'], ['vi']],
        'I(AC)': [['V', 'I(AC)', 'I(AC)']],
        'V(HC)': [['V(HC)']],
        'iii': [['iii']],
    }

    Contour = {
        '$': [['H']],
        'U': [['U', 'H'], ['H', 'U']],
        'H': [['U', 'D'], ['D', 'U']],
        'D': [['D', 'H'], ['H', 'D']],
        '_': [['_']],
    }

    Rhythm = {
        # top
        '$': [['A', 'B'], ],
        # section
        'A': [['presentation', 'continuation']],
        'B': [['antecedent', 'consequent']],
        # phrase
        'presentation': [['basic_idea', 'basic_idea\'']],
        'continuation': [['cont', 'cadence']],
        'antecedent': [['basic_idea', 'contrasting_idea']],
        'consequent': [['basic_idea\'', 'contrasting_idea\'']],
        # sub-phrase
        'basic_idea': [['bi_bar', 'bi_bar']],
        'basic_idea\'': [['bi_bar', 'bi_bar']],
        'cont': [['cont_bar', 'cont_bar']],
        'cadence': [['cont_bar', 'cadence_bar']],
        'contrasting_idea':[['bi_bar', 'bi_bar']],
        'contrasting_idea\'':[['bi_bar', 'cadence_bar']],
        # bar
        'bi_bar': [
            [['e', 'e'], ['e','e'], ['e', 'e']],
            [['s','s','s','s'],['e', 'e'],['e', 'e']],
            [['s', 's', 's', 's'], ['s', 's', 's', 's'], ['s', 's', 's', 's']],
        ],
        'cont_bar': [[['e', 'e'], ['e', 'e'], ['e', 'e'], ],
                     [['s', 's', 's', 's'], ['s', 's', 's', 's'], ['s', 's', 's', 's']]],
        'cadence_bar': [[['s', 's', 's', 's'], 'q', 'r_q'], [['e', 's', 's'], 'q', 'r_q'], [['e', 'e'], 'q', 'r_q']],
        # note
        'h': [['h']],
        'q': [['q']],
        'e': [['e']],
        'r_q': [['r_q']],
        # 'edot':[['edot']],
        's': [['s']]
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
