import random

from melody import Tree, Melody,Note
import copy

class Form(Tree):
    def __init__(self, symbol_cat='A', rhythm_cat=8, latent_variables=None):
        super().__init__()
        self.symbol_cat = symbol_cat
        self.rhythm_cat = rhythm_cat
        if latent_variables is None:
            self.latent_variables = {'harmony': 'I', 'scale': [0, 2, 4, 5, 7, 9, 11]}
        else:
            self.latent_variables = latent_variables

    def to_melody_templates(self):
        melody_templates = []
        form_surface = copy.deepcopy(self.get_surface())
        for form in form_surface:
            pitch_population  = list(range(-12,13))
            sampled_pitches = random.sample(pitch_population,k=3)
            sampled_durations = [1.0,1.0,1.0]
            latent_variables = form.latent_variables
            melody = Melody()

            melody_templates.append(melody)
        return melody_templates

    def to_similarity_template(self):
        surface = self.get_surface()
        symbols = [x.symbol_cat for x in surface]
        return symbols


def build_sentence():
    scale = [0,2,4,7,9,11]
    sentence = Form(rhythm_cat=8)
    sentence.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    presentation, continuation = sentence.children
    presentation.add_children([Form(rhythm_cat=2, symbol_cat='a'), Form(rhythm_cat=2, symbol_cat='a')])
    continuation.add_children([Form(rhythm_cat=2, symbol_cat='a(frag,v+)', latent_variables={'harmony': 'V','scale':scale}),
                               Form(rhythm_cat=2, symbol_cat='c')])
    presentation.children[0].add_children([Form(rhythm_cat=1, symbol_cat='a'),
                                           Form(rhythm_cat=1, symbol_cat='b', latent_variables={'harmony': 'V','scale':scale})])
    presentation.children[1].add_children([Form(rhythm_cat=1, symbol_cat='a'),
                                           Form(rhythm_cat=1, symbol_cat='b', latent_variables={'harmony': 'V','scale':scale})])
    continuation.children[0].add_children([Form(rhythm_cat=1, symbol_cat='a(frag,v+)'),
                                           Form(rhythm_cat=1, symbol_cat='b(frag,v+)')])
    continuation.children[1].add_children([Form(rhythm_cat=1, symbol_cat='c', latent_variables={'harmony': 'V','scale':scale}),
                                           Form(rhythm_cat=1, symbol_cat='cadence')])
    return sentence


def build_period():
    period = Form(rhythm_cat=8)
    period.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    antecedent, consequent = period.children
    antecedent.add_children([Form(rhythm_cat=2, symbol_cat='a'), Form(rhythm_cat=2, symbol_cat='-a')])
    consequent.add_children([Form(rhythm_cat=2, symbol_cat='a\'', latent_variables={'harmony': 'V'}),
                             Form(rhythm_cat=2, symbol_cat='c')])
    antecedent.children[0].add_children([Form(rhythm_cat=1, symbol_cat='a'),
                                         Form(rhythm_cat=1, symbol_cat='b', latent_variables={'harmony': 'V'})])
    antecedent.children[1].add_children([Form(rhythm_cat=1, symbol_cat='-a', latent_variables={'harmony': 'V'}),
                                         Form(rhythm_cat=1, symbol_cat='-b', latent_variables={'harmony': 'I'})])
    consequent.children[0].add_children([Form(rhythm_cat=1, symbol_cat='a\''),
                                         Form(rhythm_cat=1, symbol_cat='b\'')])
    consequent.children[1].add_children([Form(rhythm_cat=1, symbol_cat='c', latent_variables={'harmony': 'V'}),
                                         Form(rhythm_cat=1, symbol_cat='cadence')])
    return period


if __name__ == '__main__':
    sentence = build_sentence()
    period = build_period()
    print('\n*** sentence ***\n')
    sentence.show()
    print('\n*** period ***\n')
    period.show()

    print('\n--- similarity template ---\n')
    print(sentence.to_similarity_template())
    melody_templates = sentence.to_melody_templates()
    for melody_template in melody_templates:
        melody_template.show()
