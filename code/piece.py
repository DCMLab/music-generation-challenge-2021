import copy
from templates import Grammar


def chunks(lst, n):
    l = len(lst) // n
    remainder = len(lst) % n
    if n > len(lst):

        tiled_list = sum([[ele for _ in range(n // len(lst)+1) ]for ele in lst],[])

        distributed_lst = chunks(tiled_list,n)
    elif remainder == 0:
        distributed_lst = [lst[i*l:(i+1)*l] for i in range(n)]

    else:
        ## remainder * size l+1 + (n-remainder) of size l
        longer_part = [lst[i*(l+1):(i+1)*(l+1)] for i in range(remainder)]
        shorter_part = [lst[remainder*(l+1)+i*l:remainder*(l+1)+(i+1)*l] for i in range(n-remainder)]
        distributed_lst = longer_part + shorter_part
    assert len(distributed_lst) == n, (lst,distributed_lst)

    return distributed_lst

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
        if type(self.char) == list:
            new_symbols = [Symbol(char) for char in self.char]
        else:
            new_chars = Grammar.make_deterministic(grammar_dict)[self.char]
            new_symbols = [Symbol(char) for char in new_chars]
        return new_symbols

    def __repr__(self):
        return str(self.char)


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
        expansion= [symbol.expand(grammar_dict=self.grammar_dict)
                               for symbol in self.symbols]
        expansions_flat = sum(expansion, [])
        assert all([isinstance(x,Symbol) for x in expansions_flat])
        expansions_word = Word(symbols=expansions_flat, grammar_dict=self.grammar_dict)

        return expansions_word

    def start(self):
        """[]->starting word"""
        expansions = self.expand()
        return expansions

    def distribute(self,n):
        partition = chunks(self.symbols, n)
        partition = [Word(symbols=x,grammar_dict=self.grammar_dict) for x in partition]
        return partition

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


def node_grow(tree: Tree, level):
    """tree is suppose to have no children"""
    if level == 'Top':
        # start Form, start Key, start & distribute Harmony,start Rhythm
        primary_component_expansion = tree.data.form.start().symbols
        n_partition = len(primary_component_expansion)
        tree.data.key = tree.data.key.start()
        tree.data.harmony = tree.data.harmony.start()
        harmony_expansion_partition = tree.data.harmony.expand_and_distribute(n=n_partition)
        rhythm_expansion_partition = tree.data.rhythm.expand_and_distribute(n=n_partition)
        for symbol,harmony,rhythm in zip(primary_component_expansion,harmony_expansion_partition,rhythm_expansion_partition):
            child = Tree()
            child.data = copy.deepcopy(tree.data)
            child.data.level = 'Section'
            child.data.form = Form(symbols=[symbol])
            child.data.rhythm = rhythm
            child.data.harmony = harmony
            tree.children.append(child)

    elif level == 'Section':
        # *expand Form, expand & distribute Harmony, start Contour, expand and distribute Rhythm
        for section in tree.children:
            primary_component_expansion = section.data.form.expand().symbols
            n_partition = len(primary_component_expansion)
            harmony_expansion_partition = section.data.harmony.expand_and_distribute(n=n_partition)
            rhythm_expansion_partition = section.data.rhythm.expand_and_distribute(n=n_partition)
            for symbol,harmony,rhythm in zip(primary_component_expansion,harmony_expansion_partition,rhythm_expansion_partition):

                child = Tree()
                child.data = copy.deepcopy(section.data)
                child.data.level = 'Phrase'
                child.data.form = Form(symbols=[symbol])
                child.data.harmony = harmony
                child.data.contour = child.data.contour.start()
                child.data.rhythm = rhythm
                section.children.append(child)

    elif level == 'Phrase':
        # phrase level: *expand Form, expand & distribute Harmony, expand & distribute Contour, expand & distribute Rhythm, start GuideTone
        for section in tree.children:
            for phrase in section.children:
                primary_component_expansion = phrase.data.form.expand().symbols
                n_partition = len(primary_component_expansion)
                harmony_expansion_partition = phrase.data.harmony.expand_and_distribute(n=n_partition)
                contour_expansion_partition = phrase.data.contour.expand_and_distribute(n=n_partition)
                rhythm_expansion_partition = phrase.data.rhythm.expand_and_distribute(n=n_partition)

                for form, harmony, contour,rhythm in zip(primary_component_expansion, harmony_expansion_partition,
                                                  contour_expansion_partition,rhythm_expansion_partition):
                    child = Tree()
                    child.data = copy.deepcopy(phrase.data)
                    child.data.level = 'Subphrase'
                    child.data.form = Form(symbols=[form])
                    child.data.harmony = harmony
                    child.data.contour = contour
                    child.data.rhythm = rhythm
                    child.data.guide_tone = child.data.guide_tone.start()
                    phrase.children.append(child)

    elif level == 'Subphrase':
        # Subphrase level: expand Rhythm expand_distribute Form, expand & distribute Harmony, expand & distribute Contour,
        for section in tree.children:
            for phrase in section.children:
                for subphrase in phrase.children:
                    primary_component_expansion = subphrase.data.rhythm.expand().symbols
                    n_partition = len(primary_component_expansion)
                    harmony_expansion_partition = subphrase.data.harmony.expand_and_distribute(n=n_partition)
                    contour_expansion_partition = subphrase.data.contour.expand_and_distribute(n=n_partition)
                    form_expansion_partition = subphrase.data.form.expand_and_distribute(n=n_partition)
                    #print(form_expansion_partition)
                    for rhythm, harmony, contour, form in zip(primary_component_expansion, harmony_expansion_partition,
                                                              contour_expansion_partition, form_expansion_partition):
                        child = Tree()
                        child.data = copy.deepcopy(subphrase.data)
                        child.data.level = 'Bar'
                        child.data.rhythm = Rhythm(symbols=[rhythm])
                        child.data.harmony = harmony
                        child.data.contour = contour
                        child.data.form = form
                        subphrase.children.append(child)



    elif level == 'Bar':
        # bar level: *expand Rhythm, Distribute Harmony, Expand & Distribute Contour
        for section in tree.children:
            for phrase in section.children:
                for subphrase in phrase.children:
                    for bar in subphrase.children:
                        primary_component_expansion = bar.data.rhythm.expand().symbols
                        n_partition = len(primary_component_expansion)
                        harmony_expansion_partition = bar.data.harmony.distribute(n=n_partition)
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
        # beat level: *expand rhythm, expand & distribute contour
        for section in tree.children:
            for phrase in section.children:
                for subphrase in phrase.children:
                    for bar in subphrase.children:
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

    node_grow(tree=test_tree, level='Top')
    node_grow(tree=test_tree, level='Section')
    node_grow(tree=test_tree, level='Phrase')
    node_grow(tree=test_tree, level='Subphrase')
    node_grow(tree=test_tree, level='Bar')
    node_grow(tree=test_tree, level='Beat')
    print(test_tree)
    from tree_to_xml import tree_to_stream_powerful

    stream = tree_to_stream_powerful(test_tree)
    stream.show()

