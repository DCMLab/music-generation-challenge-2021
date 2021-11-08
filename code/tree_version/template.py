import copy
import random
from form import melody_templates, similarity_template
from melody import Melody, Note


def old_template_to_tree(padded_old_template):
    melody = padded_old_template['melody']
    latent_variables = padded_old_template['latent_info']
    harmony = latent_variables['harmony']

    tree = Melody('root', latent_variables=latent_variables)
    old_melody_head = melody['head']
    old_melody_body = melody['body']
    old_melody_tail = melody['tail']
    old_melody = old_melody_head + old_melody_body + old_melody_tail
    old_harmony_head = harmony['head']
    old_harmony_body = harmony['body']
    old_harmony_tail = harmony['tail']
    old_harmony = old_harmony_head + old_harmony_body + old_harmony_tail

    for i, (x, h) in enumerate(zip(old_melody_head, old_harmony_head)):
        if x == '_':
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (old_harmony_head[i - 1], old_harmony_head[i + 1])
            new_latent_variables['harmony'] = harmony_transition
            left_pitch = old_melody_head[i - 1]
            right_pitch = old_melody_head[i + 1]
            pitch_transition = (left_pitch, right_pitch)
            rhythm_transition = (left_rhythm, right_rhythm)
            tree.add_children([Melody(pitch_transition, latent_variables=new_latent_variables, part='head')])

    for i, (x, h) in enumerate(zip(old_melody_body, old_harmony_body)):
        if x == '_':
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (old_harmony_body[i - 1], old_harmony_body[i + 1])
            new_latent_variables['harmony'] = harmony_transition
            left_pitch = old_melody_body[i - 1]
            right_pitch = old_melody_body[i + 1]
            pitch_transition = (left_pitch, right_pitch)
            tree.add_children([Melody(pitch_transition, latent_variables=new_latent_variables, part='body')])

    for i, (x, h) in enumerate(zip(old_melody_tail, old_harmony_tail)):
        if x == '_':
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (old_harmony_tail[i - 1], old_harmony_tail[i + 1])
            new_latent_variables['harmony'] = harmony_transition
            left_pitch = old_melody_tail[i - 1]
            right_pitch = old_melody_tail[i + 1]
            pitch_transition = (left_pitch, right_pitch)
            tree.add_children([Melody(pitch_transition, latent_variables=new_latent_variables, part='tail')])
    return tree


def make_harmony_same_format_as_melody(old_temp):
    new_old_temp = copy.deepcopy(old_temp)
    latent_variables = new_old_temp['latent_info']
    old_melody = new_old_temp['melody']
    harmony = latent_variables['harmony']
    if type(harmony[0]) != list:
        converted_harmony = []
        for x in old_melody:
            if x == '_':
                converted_harmony.append('_')
            else:
                converted_harmony.append(harmony)
        new_old_temp['latent_info']['harmony'] = converted_harmony
    return new_old_temp


def add_head_or_tail(old_templates):
    all_start_and_end_notes = [[old_temp['melody'][0], old_temp['melody'][-1]] for old_temp in old_templates]
    all_start_and_end_notes = [['start', 'start']] + all_start_and_end_notes + [['end', 'end']]
    all_start_and_end_harmony = [[old_temp['latent_info']['harmony'][0], old_temp['latent_info']['harmony'][-1]] for
                                 old_temp in old_templates]
    all_start_and_end_harmony = [['start', 'start']] + all_start_and_end_harmony + [['end', 'end']]
    padded_old_temps = []
    last_bar_has_tail = False
    for i, old_temp in enumerate(old_templates):
        padded_old_temp = copy.deepcopy(old_temp)
        melody_head = [all_start_and_end_notes[1 + i - 1][1], '_', all_start_and_end_notes[1 + i][0]]
        melody_tail = [all_start_and_end_notes[1 + i][1], '_', all_start_and_end_notes[1 + i + 1][0]]
        harmony_head = [all_start_and_end_harmony[1 + i - 1][1], '_', all_start_and_end_harmony[1 + i][0]]
        harmony_tail = [all_start_and_end_harmony[1 + i][1], '_', all_start_and_end_harmony[1 + i + 1][0]]
        if last_bar_has_tail:
            add_what = random.choice(['none'])
            # add_what = random.choice(['tail', 'none'])
        else:
            add_what = random.choice(['none'])
            # add_what = random.choice(['head', 'tail', 'head_and_tail'])
        # print('add_what: ',add_what)
        if add_what == 'head':
            last_bar_has_tail = False
            padded_old_temp['melody'] = {'head': melody_head, 'body': padded_old_temp['melody'], 'tail': []}
            padded_old_temp['duration'] = {'head': melody_head, 'body': padded_old_temp['melody'], 'tail': []}
            padded_old_temp['latent_info']['harmony'] = {'head': harmony_head,
                                                         'body': padded_old_temp['latent_info']['harmony'],
                                                         'tail': []}
        elif add_what == 'tail':
            last_bar_has_tail = True
            padded_old_temp['melody'] = {'head': [], 'body': padded_old_temp['melody'], 'tail': melody_tail}
            padded_old_temp['latent_info']['harmony'] = {'head': [],
                                                         'body': padded_old_temp['latent_info']['harmony'],
                                                         'tail': harmony_tail}
        elif add_what == 'head_and_tail':
            last_bar_has_tail = True
            padded_old_temp['melody'] = {'head': melody_head, 'body': padded_old_temp['melody'], 'tail': melody_tail}
            padded_old_temp['latent_info']['harmony'] = {'head': harmony_head,
                                                         'body': padded_old_temp['latent_info']['harmony'],
                                                         'tail': harmony_tail}
        elif add_what == 'none':
            last_bar_has_tail = False
            padded_old_temp['melody'] = {'head': [], 'body': padded_old_temp['melody'], 'tail': []}
            padded_old_temp['latent_info']['harmony'] = {'head': [],
                                                         'body': padded_old_temp['latent_info']['harmony'],
                                                         'tail': []}

        padded_old_temps.append(padded_old_temp)
    return padded_old_temps


beginning = Melody()
scale = [0, 2, 4, 5, 7, 9, 11]
latent_variables = {'harmony': [0, 4, 7], 'scale': scale}
beginning.add_children(
    [Melody(transition=(
    Note(12, 1.0, latent_variables=latent_variables), Note(12 + 4, 1.0, latent_variables=latent_variables))),
     Melody(transition=(
     Note(12 + 4, 1.0, latent_variables=latent_variables), Note(12, 1.0, latent_variables=latent_variables)))
     ])

second = Melody()
second.add_children(
    [Melody(transition=(Note(11, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
                        Note(9, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(9, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
                        Note(7, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(7, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
                        Note(5, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(5, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
                        Note(4, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(4, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}),
                        Note(2, 0.5, latent_variables={'harmony': [2, 7, 11], 'scale': scale}))),
     ])

seq_1 = Melody()
seq_1.add_children([Melody(
    transition=(Note(-5, 1.0, latent_variables=latent_variables), Note(7, 1.0, latent_variables=latent_variables))),
    Melody(transition=(Note(7, 1.0, latent_variables=latent_variables),
                       Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale})))
])

seq_2 = Melody()
seq_2.add_children(
    [Melody(transition=(Note(-5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(4, 1.0, latent_variables=latent_variables)))
     ])

pre_cadence = Melody()
pre_cadence.add_children(
    [Melody(transition=(Note(0, 1.0, latent_variables=latent_variables),
                        Note(7, 1.0, latent_variables=latent_variables))),
     Melody(transition=(Note(7, 1.0, latent_variables=latent_variables),
                        Note(7, 1.0, latent_variables=latent_variables)))
    ])

cadence = Melody(no_tail=True)
cadence.add_children(
    [Melody(transition=(Note(4, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(0, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False))),
     Melody(transition=(Note(0, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False),
                        Note(-12, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False)))
     ])

beginning2 = Melody()
beginning2.add_children(
    [Melody(transition=(Note(12 + 4, 1.0, latent_variables=latent_variables),
                        Note(7, 1.0, latent_variables=latent_variables))),
     Melody(transition=(Note(7, 1.0, latent_variables=latent_variables),
                        Note(0, 1.0, latent_variables=latent_variables)))
     ])

second2 = Melody()
second2.add_children(
    [Melody(transition=(Note(12 + 2, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(7, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(7, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(-1, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale})))
     ])

third2 = Melody()
third2.add_children(
    [Melody(transition=(Note(12, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': scale}),
                        Note(9, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': scale}))),
     Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': scale}),
                        Note(5, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': scale})))
     ])

half_cadence = Melody(no_tail=True)
half_cadence.add_children(
    [Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': scale}),
                        Note(7, 2.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False))),
     ])

beginning3 = Melody()
beginning3.add_children(
    [Melody(transition=(Note(0, 1.0, latent_variables={'harmony': [0, 4, 7, 10], 'scale': [0, 2, 4, 5, 7, 9, 10]}),
                        Note(7, 1.0, latent_variables={'harmony': [0, 4, 7, 10], 'scale': [0, 2, 4, 5, 7, 9, 10]}))),
     Melody(transition=(Note(7, 1.0, latent_variables={'harmony': [0, 4, 7, 10], 'scale': [0, 2, 4, 5, 7, 9, 10]}),
                        Note(10, 1.0, latent_variables={'harmony': [0, 4, 7, 10], 'scale': [0, 2, 4, 5, 7, 9, 10]})))
     ])

second3 = Melody(no_tail=True)
second3.add_children(
    [Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': [0, 2, 4, 5, 7, 9, 10]}),
                        Note(4, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': [0, 2, 4, 5, 7, 9, 10]}))),
     Melody(transition=(Note(4, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': [0, 2, 4, 5, 7, 9, 10]}),
                        Note(5, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': [0, 2, 4, 5, 7, 9, 10]},time_stealable=False)))
     ])

beginning4 = Melody()
beginning4.add_children(
    [Melody(transition=(Note(2, 1.0, latent_variables={'harmony': [2, 5, 9], 'scale': scale}),
                        Note(9, 1.0, latent_variables={'harmony': [2, 5, 9], 'scale': scale}))),
     Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [2, 5, 9], 'scale': scale}),
                        Note(12, 1.0, latent_variables={'harmony': [2, 5, 9], 'scale': scale})))
     ])

second4 = Melody(no_tail=True)
second4.add_children(
    [Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(7, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False))),
     ])

beginning5 = Melody()
beginning5.add_children(
    [Melody(transition=(Note(12, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale}),
                        Note(9, 1.0, latent_variables={'harmony': [0, 4, 9], 'scale': scale}))),
     Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [0, 4, 9], 'scale': scale}),
                        Note(5, 1.0, latent_variables={'harmony': [2, 5, 9], 'scale': scale})))
     ])

second5 = Melody()
second5.add_children(
    [Melody(transition=(Note(9, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': scale}),
                        Note(7, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}))),
     Melody(transition=(Note(7, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(4, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale})))
     ])

pre_cadence2 = Melody()
pre_cadence2.add_children(
    [Melody(transition=(Note(-5, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale}),
                        Note(7, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale}))),
     Melody(transition=(Note(7, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale}),
                        Note(7, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale})))
     ])

cadence_final = Melody(no_tail=True)
cadence_final.add_children(
    [Melody(transition=(Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}),
                        Note(-1, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False))),
     Melody(transition=(Note(-1, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale},time_stealable=False),
                        Note(0, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale},time_stealable=False)))
     ])

tree_templates = [beginning,
                  second,
                  seq_1,
                  seq_2,
                  copy.deepcopy(seq_1),
                  copy.deepcopy(seq_2),
                  pre_cadence,
                  cadence,
                  ##
                  beginning2,
                  second2,
                  third2,
                  half_cadence,
                  ##
                  beginning3,
                  second3,
                  beginning4,
                  second4,
                  ##
                  beginning5,
                  second5,
                  pre_cadence2,
                  cadence_final
                  ]

handcoded_similarity = ['a', 'b', 'c', 'c', 'c', 'c', 'd', 'e',
                        'f', 'f', 'f', 'g',
                        'h', 'i', 'h', 'i',
                        'j', 'j', 'd', 'e']


def pad_melody_templates(melody_templates=list[Melody]) -> list[Melody]:
    # melody_templates = copy.deepcopy(_melody_templates)
    last_bar_has_tail = False
    self_similarity_template = handcoded_similarity
    memory_padding = {}
    assert len(melody_templates) == len(self_similarity_template)
    for i, (melody_template, symbol) in enumerate(zip(melody_templates, self_similarity_template)):
        if symbol in memory_padding.keys():
            add_what = memory_padding[symbol]
        else:
            if last_bar_has_tail or i == 0:
                if i != len(melody_templates) - 1:
                    choices = ['none'] + (not melody_template.no_tail) * ['tail']
                    print('\n\n{}{}!!!!! choices: {}\n\n'.format(i, not melody_template.no_tail, choices))
                    add_what = random.choice(choices)
                else:
                    add_what = 'none'
            else:
                if i != len(melody_templates) - 1:
                    choices = ['head', 'none'] + (not melody_template.no_tail) * ['head_and_tail', 'tail']
                    print('\n\n{}{}!!!!! choices: {}\n\n'.format(i, not melody_template.no_tail, choices))
                    add_what = random.choice(choices)
                else:
                    add_what = random.choice(['head', 'none'])
            memory_padding[symbol] = add_what
            print('add_what: ', add_what)
        # add corresponding subtrees to head or tail
        if add_what == 'head':

            previous_bar = melody_templates[i - 1]
            previous_note = copy.deepcopy(previous_bar.children[-1].transition[1])
            first_note = melody_template.children[0].transition[0]
            new_transition = (previous_note, first_note)
            added_head = Melody(transition=new_transition, part='head')
            added_head.parent = melody_template
            melody_template.children = [added_head] + melody_template.children
        elif add_what == 'tail':
            print('before padding:', len(melody_template.children))
            last_bar_has_tail = True
            next_bar = melody_templates[i + 1]
            next_note = copy.deepcopy(next_bar.children[0].transition[0])
            last_note = copy.deepcopy(melody_template.children[-1].transition[1])
            print('pitch_cat of transition:', last_note.pitch_cat, next_note.pitch_cat)
            new_transition = (last_note, next_note)
            melody_template.add_children([Melody(transition=new_transition, part='tail')])
            print('after padding:', len(melody_template.children))
        elif add_what == 'head_and_tail':

            last_bar_has_tail = True
            next_bar = melody_templates[i + 1]
            next_note = copy.deepcopy(next_bar.children[0].transition[0])
            last_note = copy.deepcopy(melody_template.children[-1].transition[1])
            tail_transition = (last_note, next_note)
            previous_bar = melody_templates[i - 1]
            previous_note = previous_bar.children[-1].transition[1]
            first_note = melody_template.children[0].transition[0]
            head_transition = (previous_note, first_note)
            added_head = Melody(transition=head_transition, part='head')
            added_head.parent = melody_template
            melody_template.add_children([Melody(transition=tail_transition, part='tail')])
        elif add_what == 'none':
            pass
        else:
            assert False, add_what

    return melody_templates


padded_melody_templates = pad_melody_templates(tree_templates)
for x in padded_melody_templates:
    print('************')
    x.show()

# padded_old_templates = add_head_or_tail(old_templates=piece_old_templates)


# tree_templates = [old_template_to_tree(x) for x in padded_old_templates]

# print(tree_templates)
if __name__ == '__main__':
    pass
