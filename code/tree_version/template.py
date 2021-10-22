import copy
import random

from melody import Melody


def old_template_to_tree(padded_old_template):
    melody = padded_old_template['melody']
    latent_variables = padded_old_template['latent_info']
    harmony = latent_variables['harmony']

    tree = Melody('root', latent_variables=latent_variables)
    old_melody_head = melody['head']
    old_melody_body = melody['body']
    old_melody_tail = melody['tail']
    old_melody = old_melody_head+old_melody_body+old_melody_tail
    old_harmony_head = harmony['head']
    old_harmony_body = harmony['body']
    old_harmony_tail = harmony['tail']
    old_harmony = old_harmony_head+old_harmony_body+old_harmony_tail

    for i, (x, h) in enumerate(zip(old_melody_head, old_harmony_head)):
        if x == '_':
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (old_harmony_head[i - 1], old_harmony_head[i + 1])
            new_latent_variables['harmony'] = harmony_transition
            left_pitch = old_melody_head[i-1]
            right_pitch = old_melody_head[i+1]
            pitch_transition = (left_pitch, right_pitch)
            tree.add_children([Melody(pitch_transition, latent_variables=new_latent_variables,part='head')])

    for i, (x, h) in enumerate(zip(old_melody_body, old_harmony_body)):
        if x == '_':
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (old_harmony_body[i - 1], old_harmony_body[i + 1])
            new_latent_variables['harmony'] = harmony_transition
            left_pitch = old_melody_body[i-1]
            right_pitch = old_melody_body[i+1]
            pitch_transition = (left_pitch, right_pitch)
            tree.add_children([Melody(pitch_transition, latent_variables=new_latent_variables,part='body')])

    for i, (x, h) in enumerate(zip(old_melody_tail, old_harmony_tail)):
        if x == '_':
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (old_harmony_tail[i - 1], old_harmony_tail[i + 1])
            new_latent_variables['harmony'] = harmony_transition
            left_pitch = old_melody_tail[i-1]
            right_pitch = old_melody_tail[i+1]
            pitch_transition = (left_pitch, right_pitch)
            tree.add_children([Melody(pitch_transition, latent_variables=new_latent_variables,part='tail')])
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


beginning = {
    'melody': [12, '_', 12, '_', 12],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

beginning_tree = Melody('root', latent_variables=beginning['latent_info'])
beginning_tree.add_children([Melody((12, 12), latent_variables=beginning['latent_info']),
                             Melody((12, 12), latent_variables=beginning['latent_info'])])

second = {
    'melody': [11, '_', 9, '_', 7, '_', 5, '_', 4, '_', 2],
    'latent_info': {
        'harmony': [2, 7, 11],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

second_tree = Melody('root', latent_variables=beginning['latent_info'])
second_tree.add_children([Melody((11, 9), latent_variables=beginning['latent_info']),
                          Melody((9, 7), latent_variables=beginning['latent_info']),
                          Melody((7, 5), latent_variables=beginning['latent_info']),
                          Melody((5, 4), latent_variables=beginning['latent_info']),
                          Melody((4, 2), latent_variables=beginning['latent_info']),
                          ])

seq_1 = {
    'melody': [-5, '_', 7, '_', 5],
    'latent_info': {
        'harmony': [[0, 4, 7], '_', [0, 4, 7], '_', [2, 5, 7, 11]],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

seq_2 = {
    'melody': [-5, '_', 5, '_', 4],
    'latent_info': {
        'harmony': [[2, 5, 7, 11], '_', [2, 5, 7, 11], '_', [0, 4, 7]],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

pre_cadence = {
    'melody': [0, '_', 7, '_', 7],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

cadence = {
    'melody': [4, '_', 2, '_', 0],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

piece_old_templates = [beginning,
                       second,
                       seq_1,
                       seq_2,
                       seq_1,
                       seq_2,
                       pre_cadence,
                       cadence]


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
        melody_head = [all_start_and_end_notes[1 + i - 1][1], '_',all_start_and_end_notes[1 + i][0]]
        melody_tail = [all_start_and_end_notes[1 + i][1],'_', all_start_and_end_notes[1 + i + 1][0]]
        harmony_head = [all_start_and_end_harmony[1 + i - 1][1], '_',all_start_and_end_harmony[1 + i][0]]
        harmony_tail = [all_start_and_end_harmony[1 + i][1],'_', all_start_and_end_harmony[1 + i + 1][0]]
        if last_bar_has_tail:
            add_what = random.choice(['none'])
            #add_what = random.choice(['tail', 'none'])
        else:
            add_what = random.choice(['none'])
            #add_what = random.choice(['head', 'tail', 'head_and_tail'])
        #print('add_what: ',add_what)
        if add_what == 'head':
            last_bar_has_tail = False
            padded_old_temp['melody'] = {'head': melody_head, 'body': padded_old_temp['melody'], 'tail': []}
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



#print(piece_old_templates)
piece_old_templates = [make_harmony_same_format_as_melody(old_temp=old_temp) for old_temp in piece_old_templates]
#print('piece_old_templates: ',piece_old_templates)
padded_old_templates = add_head_or_tail(old_templates=piece_old_templates)
#print('padded_old_templates: ', padded_old_templates)

tree_templates = [old_template_to_tree(x) for x in padded_old_templates]
#print(tree_templates)
if __name__ == '__main__':

    for template in tree_templates:
        print('----')
        template.show()
        for child in template.children:
            pass
            #print('value: ',child.value)
            print('part: ', child.part, 'latent_variables: ', child.latent_variables)
