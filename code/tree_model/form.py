import itertools
import random

from melody import Tree, Melody, Note
import copy


class Form(Tree):
    def __init__(self, symbol_cat='A', max_elaboration=3, rhythm_cat=8, latent_variables=None, no_tail=False,
                 time_stealable=True, repeat_type=None):
        super().__init__()
        self.symbol_cat = symbol_cat
        self.rhythm_cat = rhythm_cat
        self.max_elaboration = max_elaboration
        self.no_tail = no_tail
        self.time_stealable = time_stealable
        self.repeat_type = repeat_type
        if latent_variables is None:
            self.latent_variables = {'harmony': [0, 4, 7], 'scale': [0, 2, 4, 5, 7, 9, 11]}
        else:
            self.latent_variables = latent_variables

    def to_melody_templates(self):
        melody_templates = []
        form_surface = copy.deepcopy(self.get_surface())
        symbol_dict = {}
        chord_degree_dict = {}
        for form in form_surface:

            latent_variables = form.latent_variables
            if form.symbol_cat == 'HC':
                if {2, 7, 11}.issubset(form.latent_variables['harmony']):
                    melody = Melody(no_tail=form.no_tail, max_elaboration=4, repeat_type=form.repeat_type)

                    melody.add_children([
                        Melody(
                            transition=(Note(pitch_cat=12 + 5, rhythm_cat=0.5, latent_variables=latent_variables),
                                        Note(pitch_cat=12 + 4, rhythm_cat=0.5, latent_variables=latent_variables))),
                        Melody(
                            transition=(Note(pitch_cat=12 + 4, rhythm_cat=0.5, latent_variables=latent_variables),
                                        Note(pitch_cat=12 + 2, rhythm_cat=1.0, latent_variables=latent_variables,
                                             time_stealable=False))),
                        Melody(transition=(
                            Note(pitch_cat=12 + 2, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False),
                            Note(pitch_cat=7, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False))),
                    ])
                elif form.latent_variables['harmony'] == [4, 8, 11]:
                    melody = Melody(no_tail=form.no_tail, max_elaboration=4, repeat_type=form.repeat_type)

                    melody.add_children([
                        Melody(
                            transition=(Note(pitch_cat=12 + 2, rhythm_cat=0.5,
                                             latent_variables={'harmony': [4, 8, 11],
                                                               'scale': latent_variables['scale']}),
                                        Note(pitch_cat=12, rhythm_cat=0.5, latent_variables={'harmony': [4, 8, 11],
                                                                                             'scale':
                                                                                                 latent_variables[
                                                                                                     'scale']}))),
                        Melody(
                            transition=(Note(pitch_cat=12, rhythm_cat=0.5, latent_variables={'harmony': [4, 8, 11],
                                                                                             'scale':
                                                                                                 latent_variables[
                                                                                                     'scale']}),
                                        Note(pitch_cat=11, rhythm_cat=1.0, latent_variables=latent_variables,
                                             time_stealable=False))),
                        Melody(transition=(
                            Note(pitch_cat=11, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False),
                            Note(pitch_cat=4, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False))),
                    ])
                else:
                    assert False
            elif form.symbol_cat == 'PAC':
                if form.latent_variables['harmony'] == [0, 4, 7]:
                    melody = Melody(no_tail=form.no_tail, max_elaboration=4, repeat_type=form.repeat_type)
                    melody.add_children([
                        Melody(
                            transition=(Note(pitch_cat=12 + 4, rhythm_cat=0.5, latent_variables=latent_variables),
                                        Note(pitch_cat=12 + 2, rhythm_cat=0.5, latent_variables=latent_variables))),
                        Melody(
                            transition=(Note(pitch_cat=12 + 2, rhythm_cat=0.5, latent_variables=latent_variables),
                                        Note(pitch_cat=12, rhythm_cat=1.0, latent_variables=latent_variables,
                                             time_stealable=False))),
                        Melody(transition=(
                            Note(pitch_cat=12, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False),
                            Note(pitch_cat=0, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False, ),), )
                    ])
                elif form.latent_variables['harmony'] == [0, 4, 9]:
                    melody = Melody(no_tail=form.no_tail, max_elaboration=4, repeat_type=form.repeat_type)
                    melody.add_children([
                        Melody(
                            transition=(Note(pitch_cat=12, rhythm_cat=0.5, latent_variables={'harmony': [4, 8, 11],
                                                                                             'scale':
                                                                                                 latent_variables[
                                                                                                     'scale']}),
                                        Note(pitch_cat=11, rhythm_cat=0.5, latent_variables={'harmony': [4, 8, 11],
                                                                                             'scale':
                                                                                                 latent_variables[
                                                                                                     'scale']}))),
                        Melody(
                            transition=(Note(pitch_cat=11, rhythm_cat=0.5, latent_variables={'harmony': [4, 8, 11],
                                                                                             'scale':
                                                                                                 latent_variables[
                                                                                                     'scale']}),
                                        Note(pitch_cat=9, rhythm_cat=1.0, latent_variables=latent_variables,
                                             time_stealable=False))),
                        Melody(transition=(
                            Note(pitch_cat=-3, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False),
                            Note(pitch_cat=-3, rhythm_cat=1.0, latent_variables=latent_variables,
                                 time_stealable=False, ),), )])
                else:
                    assert False
            else:
                pitch_population = [x for x in list(range(-5, 12 + 12)) if
                                    x % 12 in form.latent_variables['harmony']]
                symbol_cat_origin = form.symbol_cat.partition('\'')[0]
                if symbol_cat_origin in chord_degree_dict.keys():
                    sampled_degrees = copy.deepcopy(chord_degree_dict[symbol_cat_origin])
                else:
                    guidetone_degrees_bank = [
                        [2, 3, 4],
                        [3, 4, 5],
                    ]
                    guidetone_degrees_bank = sum(
                        [list(itertools.permutations(degrees)) for degrees in guidetone_degrees_bank], [])
                    print('guidetone_degrees_bank: ', guidetone_degrees_bank)
                    sampled_degrees = random.choice(guidetone_degrees_bank)
                    print('sampled_degrees: ', sampled_degrees)
                    chord_degree_dict[symbol_cat_origin] = sampled_degrees
                # print('form.latent_variables[\'harmony\']: ',form.latent_variables['harmony'],'pitch_population: ',pitch_population,'sampled_pitches: ',sampled_pitches)
                sampled_pitches = [pitch_population[i] for i in sampled_degrees]
                sampled_durations = [1.0, 1.0, 1.0]
                melody = Melody(no_tail=form.no_tail, max_elaboration=form.max_elaboration,
                                repeat_type=form.repeat_type)
                # if latent_variables['harmony'] == [0,4,7]:
                #    V_of_latent_variables = {'harmony':[2,5,7],'scale':latent_variables['scale']}
                # else:
                #    V_of_latent_variables = latent_variables
                for i in range(2):
                    transition = (
                        Note(pitch_cat=sampled_pitches[i], rhythm_cat=sampled_durations[i],
                             latent_variables=latent_variables),
                        Note(pitch_cat=sampled_pitches[i + 1], rhythm_cat=sampled_durations[i + 1],
                             latent_variables=latent_variables, time_stealable=form.time_stealable))
                    melody.add_children([Melody(transition=transition)])
            symbol_dict[form.symbol_cat] = melody
            print('len(melody.children): ', len(melody.children))

            melody_templates.append(melody)
        # print('form symbol_dict.keys(): ',symbol_dict.keys())
        return melody_templates

    def to_similarity_template(self):
        surface = self.get_surface()
        symbols = [x.symbol_cat.replace('', '') for x in surface]
        #symbols = [x.symbol_cat.replace('\'', '') for x in surface]
        return symbols


def build_sentence():
    scale = [0, 2, 4, 5, 7, 9, 11]
    sentence = Form(rhythm_cat=8)
    sentence.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    presentation, continuation = sentence.children
    presentation.add_children([Form(rhythm_cat=2, symbol_cat='a'),
                               Form(rhythm_cat=2, symbol_cat='a')])
    continuation.add_children(
        [Form(rhythm_cat=2, symbol_cat='a(frag,v+)', latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
         Form(rhythm_cat=2, symbol_cat='c', no_tail=True)])

    presentation.children[0].add_children([Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5),
                                           Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5,
                                                latent_variables={'harmony': [2, 7, 11], 'scale': scale}, no_tail=True,
                                                time_stealable=False)])
    presentation.children[1].add_children([Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5),
                                           Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5,
                                                latent_variables={'harmony': [2, 7, 11], 'scale': scale}, no_tail=True,
                                                time_stealable=False)])
    continuation.children[0].add_children(
        [Form(rhythm_cat=1, symbol_cat='c', max_elaboration=5, latent_variables={'harmony': [0, 5, 9], 'scale': scale},
              repeat_type='|:'),
         Form(rhythm_cat=1, symbol_cat='c\'', max_elaboration=5,
              latent_variables={'harmony': [0, 4, 7], 'scale': scale})])
    continuation.children[1].add_children(
        [Form(rhythm_cat=1, symbol_cat='c\'\'', max_elaboration=5,
              latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
         Form(rhythm_cat=1, symbol_cat='PAC', no_tail=True, repeat_type=':|')])
    return sentence


def build_period():
    scale = [0, 2, 4, 5, 7, 9, 11]
    period = Form(rhythm_cat=8)
    period.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    antecedent, consequent = period.children
    antecedent.add_children([Form(rhythm_cat=2, symbol_cat='a'), Form(rhythm_cat=2, symbol_cat='HC')])
    consequent.add_children([Form(rhythm_cat=2, symbol_cat='a\''), Form(rhythm_cat=2, symbol_cat='PAC')])
    i_latent_variables = {'harmony': [0, 4, 7], 'scale': scale}
    V_latent_variables = {'harmony': [2, 7, 11], 'scale': scale}
    ii_latent_variables = {'harmony': [2, 5, 9], 'scale': scale}
    antecedent.children[0].add_children([
        Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables, repeat_type='|:'),
        Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5, no_tail=True, time_stealable=False,
             latent_variables=V_latent_variables)])
    antecedent.children[1].add_children([
        Form(rhythm_cat=1, symbol_cat='c', max_elaboration=5, latent_variables=i_latent_variables),
        Form(rhythm_cat=1, symbol_cat='HC', max_elaboration=5,
             latent_variables=V_latent_variables)])
    consequent.children[0].add_children([
        Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables, ),
        Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5, no_tail=True, time_stealable=False,
             latent_variables=ii_latent_variables)])
    consequent.children[1].add_children([
        Form(rhythm_cat=1, symbol_cat='c\'', max_elaboration=5,
             latent_variables=V_latent_variables),
        Form(rhythm_cat=1, symbol_cat='PAC', max_elaboration=5, latent_variables=i_latent_variables, repeat_type=':|')])
    return period


def build_minor_sentence():
    scale = [0, 2, 4, 5, 7, 9, 11]
    sentence = Form(rhythm_cat=8)
    sentence.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    presentation, continuation = sentence.children
    presentation.add_children([Form(rhythm_cat=2),
                               Form(rhythm_cat=2)])
    continuation.add_children(
        [Form(rhythm_cat=2),
         Form(rhythm_cat=2)])
    i_latent_variables = {'harmony': [0, 4, 9], 'scale': scale}
    V_latent_variables = {'harmony': [4, 8, 11], 'scale': scale}
    presentation.children[0].add_children(
        [Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables),
         Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5,
              latent_variables=V_latent_variables, no_tail=True,
              time_stealable=False)])
    presentation.children[1].add_children(
        [Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables),
         Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5,
              latent_variables=V_latent_variables, no_tail=True,
              time_stealable=False)])
    continuation.children[0].add_children(
        [Form(rhythm_cat=1, symbol_cat='c', max_elaboration=5, latent_variables={'harmony': [2, 5, 9], 'scale': scale},
              repeat_type='|:'),
         Form(rhythm_cat=1, symbol_cat='c\'', max_elaboration=5,
              latent_variables=i_latent_variables)])
    continuation.children[1].add_children(
        [Form(rhythm_cat=1, symbol_cat='c\'\'', max_elaboration=5,
              latent_variables=V_latent_variables),
         Form(rhythm_cat=1, symbol_cat='PAC', latent_variables=i_latent_variables, no_tail=True, repeat_type=':|')])
    return sentence

def build_minor_period():
    scale = [0, 2, 4, 5, 7, 9, 11]
    period = Form(rhythm_cat=8)
    period.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    antecedent, consequent = period.children
    antecedent.add_children([Form(rhythm_cat=2, symbol_cat='a'), Form(rhythm_cat=2, symbol_cat='HC')])
    consequent.add_children(
        [Form(rhythm_cat=2, symbol_cat='a\'', latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
         Form(rhythm_cat=2, symbol_cat='PAC')])
    i_latent_variables = {'harmony': [0, 4, 9], 'scale': scale}
    V_latent_variables = {'harmony': [4, 8, 11], 'scale': scale}
    ii_latent_variables = {'harmony': [2, 5, 11], 'scale': scale}
    antecedent.children[0].add_children([
        Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables, repeat_type='|:'),
        Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5, no_tail=True, time_stealable=False,
             latent_variables=ii_latent_variables)])
    antecedent.children[1].add_children([
        Form(rhythm_cat=1, symbol_cat='c', max_elaboration=5, latent_variables=V_latent_variables),
        Form(rhythm_cat=1, symbol_cat='HC', max_elaboration=5,
             latent_variables=V_latent_variables)])
    consequent.children[0].add_children([
        Form(rhythm_cat=1, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables, ),
        Form(rhythm_cat=1, symbol_cat='b', max_elaboration=5, no_tail=True, time_stealable=False,
             latent_variables=ii_latent_variables)])
    consequent.children[1].add_children([
        Form(rhythm_cat=1, symbol_cat='c\'', max_elaboration=5,
             latent_variables=ii_latent_variables),
        Form(rhythm_cat=1, symbol_cat='PAC', max_elaboration=5, latent_variables=i_latent_variables, repeat_type=':|')])
    return period

def build_advanced_sentence():
    scale = [0, 2, 4, 5, 7, 9, 11]
    sentence = Form(rhythm_cat=8)
    sentence.add_children([Form(rhythm_cat=4), Form(rhythm_cat=4)])
    presentation, continuation = sentence.children
    presentation.add_children([Form(rhythm_cat=2, symbol_cat='a'), Form(rhythm_cat=2, symbol_cat='HC')])
    continuation.add_children(
        [Form(rhythm_cat=2, symbol_cat='a\'', latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
         Form(rhythm_cat=2, symbol_cat='PAC')])
    i_latent_variables = {'harmony': [0, 4, 9], 'scale': scale}
    V_latent_variables = {'harmony': [4, 8, 11], 'scale': scale}
    ii_latent_variables = {'harmony': [2, 5, 11], 'scale': scale}
    iv_latent_variables = {'harmony': [2, 5, 9], 'scale': scale}
    VI_latent_variables = {'harmony': [0, 5, 9], 'scale': scale}

    presentation.children[0].add_children([
        Form(rhythm_cat=1, symbol_cat='a', max_elaboration=2, latent_variables=i_latent_variables, repeat_type='|:'),
        Form(rhythm_cat=1, symbol_cat='b', max_elaboration=4, no_tail=True, time_stealable=False,
             latent_variables=ii_latent_variables)])
    presentation.children[1].add_children([
        Form(rhythm_cat=1, symbol_cat='a', max_elaboration=2, latent_variables=V_latent_variables),
        Form(rhythm_cat=1, symbol_cat='b', max_elaboration=4, no_tail=True, time_stealable=False,
             latent_variables=i_latent_variables)])
    continuation.children[0].add_children([
        Form(rhythm_cat=1, symbol_cat='a\'1', max_elaboration=4, latent_variables=iv_latent_variables, ),
        Form(rhythm_cat=1, symbol_cat='a\'1\'1', max_elaboration=4, latent_variables=i_latent_variables)])
    continuation.children[1].add_children([
        Form(rhythm_cat=1, symbol_cat='a\'1\'2', max_elaboration=6,latent_variables=V_latent_variables),
        Form(rhythm_cat=1, symbol_cat='PAC', max_elaboration=2, latent_variables=i_latent_variables, repeat_type=':|',time_stealable=False)])

    return sentence

def get_melody_templates_and_similarity_template():
    import random
    #random.seed(1)
    Msentence = build_sentence()
    Mperiod = build_period()
    msentence = build_minor_sentence()
    mperiod = build_minor_period()

    print('\n--- melody template ---\n')
    melody_templates = msentence.to_melody_templates()
    similarity_template = msentence.to_similarity_template()
    return melody_templates,similarity_template
if __name__ == '__main__':
    pass
    # for i, melody_template in enumerate(melody_templates):
    #    print('** bar {} **'.format(i + 1))
    #    melody_template.show()
