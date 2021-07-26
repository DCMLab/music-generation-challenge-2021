import copy
import random


def chunks(lst, n):
    l = len(lst) // n
    remainder = len(lst) % n
    if n > len(lst):
        last = lst[-1]
        modified = lst + [last for _ in range(n - len(lst))]
        return chunks(modified, n)
    else:
        return [lst[i:i + l] for i in range(n - 1)] + [lst[-remainder - 1:]]


class Tree:
    def __init__(self):
        self.children = []
        self.data = SectionInfo()

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.data.asdict()) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class SectionInfo:
    """the vocabulary for the piece grammar"""

    def __init__(self):
        self.level = 'Top'
        self.form = Form(chars=['$'])
        self.key = Key(chars=['$'])
        self.harmony = Harmony(chars=['$'])
        self.contour = Contour(chars=['$'])
        self.guide_tone = GuideTone(chars=['$'])
        self.rhythm = Rhythm(chars=['$'])

    def asdict(self):
        return self.__dict__


class Symbol:
    """ examples such as A(HC), B(PAC), A(_), A(/)"""

    def __init__(self, char):
        self.char = char

    def expand(self, grammar_dict):
        new_chars = Grammar.make_deterministic(grammar_dict)[self.char]
        new_symbols = [Symbol(char) for char in new_chars]
        return new_symbols

    def __repr__(self):
        return self.char


class Word:
    """list of symbols"""

    def __init__(self, symbols, grammar_dict, chars=None):
        if symbols is None:
            symbols = [Symbol(char) for char in chars]
        self.symbols = symbols

        self.grammar_dict = grammar_dict

    def expand(self):
        """[s1,s2,s3]->[[s11,s12],[s21,s22,s23],[s31]]->[s11,s12,s21,s22,s23,s31]"""
        assert all([isinstance(x,Symbol) for x in self.symbols])
        expansions_flat = sum([symbol.expand(grammar_dict=self.grammar_dict)
                               for symbol in self.symbols], [])
        assert all([isinstance(x,Symbol) for x in expansions_flat])
        expansions_word = Word(symbols=expansions_flat, grammar_dict=self.grammar_dict)

        return expansions_word

    def start(self):
        """[]->starting word"""
        expansions = self.expand()
        return expansions

    def expand_and_distribute(self, n):
        """[s1,s2,s3]->[[s11,s12],[s21,s22,s23],[s31]]->[[s11,s12,s21,s22,s23],[s31]]
        partition the expanded word (after joining) into n subwords"""
        expansions = self.expand()
        partition = chunks(expansions.symbols, n)
        partition = [Word(symbols=x,grammar_dict=self.grammar_dict) for x in partition]
        return partition

    def __repr__(self):
        return str(self.symbols)


class Form(Word):
    def __init__(self,symbols=None,chars=None):
        super().__init__(symbols,grammar_dict=Grammar.form,chars=chars)


class Harmony(Word):
    def __init__(self,symbols=None,chars=None):
        super().__init__(symbols,grammar_dict=Grammar.Harmony,chars=chars)

class Contour(Word):
    def __init__(self,symbols=None,chars=None):
        super().__init__(symbols,grammar_dict=Grammar.Contour,chars=chars)

class Rhythm(Word):
    def __init__(self,symbols=None,chars=None):
        super().__init__(symbols,grammar_dict=Grammar.Rhythm,chars=chars)

class Key(Word):
    def __init__(self,symbols=None,chars=None):
        super().__init__(symbols,grammar_dict=Grammar.Key,chars=chars)

class GuideTone(Word):
    def __init__(self,symbols=None,chars=None):
        super().__init__(symbols,grammar_dict=Grammar.GuideTone,chars=chars)

class Grammar:
    """Symbol -> Word (or list of Words)"""
    form = {
        '$': [['A', 'B', 'A'], ],
        'A': [['a', 'a'], ['a', 'a\''], ['a\'', '_']],
        'B': [['b', 'b'], ['b', 'b\''], ['b\'', '_']],
        'C': [['c', 'c'], ['c', 'c\''], ['c\'', '_']],
        'a': [['l', 'l\''], ['l', '_']],
        'a\'': [['l', 'm\''], ['l', '_']],
        'b': [['m', 'n'], ['m', '_']],
        'b\'': [['m\'', '_'], ['m', 'm\'']],
        'c': [['n', 'n\''], ['n', '_']],
        'c\'': [['n\'', '_'], ['n', 'l']],
        '_': [['_']],
    }

    Harmony = {
        '$': [['I', 'V', 'I']],
        'I': [['I', 'V'], ['V', 'I'], ['I']],
        'V': [['ii', 'V'], ['V']],
        'IV': [['IV']],
        'ii': [['vi', 'ii'], ['ii']],
        'vi': [['iii, vi'], ['vi']],
        'I(AC)': [['I', 'I(AC)']],
        'V(HC)': [['V(HC)']],
        '|I': [['|I', 'I'], ['|I']],
    }

    Contour = {
        '$': [['H']],
        'U': [['U', 'H'], ['H', 'U']],
        'H': [['U', 'D'], ['D', 'U']],
        'D': [['D', 'H'], ['H', 'D']],
        '_': [['_']],
    }

    Rhythm = {
        '$': [['q', 'q', 'q'], ['q', 'h'], ['q', 'r_q']],
        'h':[['q','q'],['q','r_q']],
        'q': [['e', 'e'], ['edot', 's'], ['q'], ['r_q']],
        'e': [['s', 's'], ['e']],
        'r_q':[['r_q']],
        'edot':[['edot']],
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


def node_grow(tree: Tree, level):
    """tree is suppose to have no children"""
    if level == 'Top':
        # start Form, start Key
        tree.data.key = tree.data.key.start()
        for symbol in tree.data.form.start().symbols:
            child = Tree()
            child.data = copy.deepcopy(tree.data)
            child.data.level = 'Section'
            child.data.form = Form(symbols=[symbol])
            tree.children.append(child)

    elif level == 'Section':
        # *expand Form, start Harmony, start Contour
        for section in tree.children:
            for symbol in section.data.form.expand().symbols:
                child = Tree()
                child.data = copy.deepcopy(section.data)
                child.data.level = 'Phrase'
                child.data.form = Form(symbols=[symbol])
                child.data.harmony = child.data.harmony.start()
                child.data.contour = child.data.contour.start()
                section.children.append(child)

    elif level == 'Phrase':
        # phrase level: *expand Form, expand & distribute Harmony, expand & distribute Contour, start Rhythm, start GuideTone
        for section in tree.children:
            for phrase in section.children:
                primary_component_expansion = phrase.data.form.expand().symbols
                n_partition = len(primary_component_expansion)
                harmony_expansion_partition = phrase.data.harmony.expand_and_distribute(n=n_partition)
                contour_expansion_partition = phrase.data.contour.expand_and_distribute(n=n_partition)
                for form, harmony, contour in zip(primary_component_expansion, harmony_expansion_partition,
                                                  contour_expansion_partition):
                    child = Tree()
                    child.data = copy.deepcopy(phrase.data)
                    child.data.level = 'Bar'
                    child.data.form = Form(symbols=[form])
                    child.data.harmony = harmony
                    child.data.contour = contour
                    child.data.rhythm = child.data.rhythm.start()
                    child.data.guide_tone = child.data.guide_tone.start()
                    phrase.children.append(child)

    elif level == 'Bar':
        # bar level: *expand Rhythm, Expand & Distribute Harmony, Expand & Distribute Contour
        for section in tree.children:
            for phrase in section.children:
                for bar in phrase.children:
                    primary_component_expansion = bar.data.rhythm.expand().symbols
                    n_partition = len(primary_component_expansion)
                    harmony_expansion_partition = bar.data.harmony.expand_and_distribute(n=n_partition)
                    contour_expansion_partition = bar.data.contour.expand_and_distribute(n=n_partition)
                    for rhythm, harmony, contour in zip(primary_component_expansion,
                                                        harmony_expansion_partition,
                                                        contour_expansion_partition):
                        child = Tree()
                        child.data = copy.deepcopy(bar.data)
                        child.data.level = 'Beat'
                        child.data.rhythm = Rhythm(symbols=[rhythm])
                        child.data.harmony = harmony
                        child.data.contour = contour
                        bar.children.append(child)

    elif level == 'Beat':
        # beat level: *expand rhythm, expand & distribute expanded contour
        for section in tree.children:
            for phrase in section.children:
                for bar in phrase.children:
                    for beat in bar.children:
                        primary_component_expansion = beat.data.rhythm.expand().symbols
                        n_partition = len(primary_component_expansion)
                        contour_expansion_partition = beat.data.contour.expand_and_distribute(n=n_partition)
                        for rhythm, contour in zip(primary_component_expansion, contour_expansion_partition):
                            child = Tree()
                            child.data = copy.deepcopy(beat.data)
                            child.data.level = 'Note'
                            child.data.rhythm = Rhythm(symbols=[rhythm])
                            child.data.contour = contour
                            beat.children.append(child)

    elif level == 'Note':
        # note level: determine note
        pass



if __name__ == '__main__':
    test_tree = Tree()
    print('top:\n',test_tree)
    node_grow(tree=test_tree, level='Top')
    print('section:\n',test_tree)
    node_grow(tree=test_tree, level='Section')
    print('phrase:\n',test_tree)
    node_grow(tree=test_tree, level='Phrase')
    print('bar:\n',test_tree)
    node_grow(tree=test_tree, level='Bar')
    print('beat:\n',test_tree)
    node_grow(tree=test_tree,level='Beat')
    print('note:\n',test_tree)