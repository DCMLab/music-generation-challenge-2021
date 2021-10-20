import copy

from melody import Melody


def old_template_to_tree(old_template):
    latent_variables = old_template['latent_info']
    tree = Melody('root',latent_variables=latent_variables)
    old_melody = old_template['melody']
    harmony = copy.deepcopy(latent_variables['harmony'])
    if type(harmony[0]) != list:
        converted_harmony = []
        for x in old_melody:
            if x =='_':
                converted_harmony.append('_')
            else:
                converted_harmony.append(harmony)
        harmony = converted_harmony
    for i,(x,h) in enumerate(zip(old_melody,harmony)):
        if x=='_':
            left_pitch = old_melody[i-1]
            right_pitch = old_melody[i + 1]
            new_latent_variables = copy.deepcopy(latent_variables)
            harmony_transition = (harmony[i-1],harmony[i+1])
            new_latent_variables['harmony'] = harmony_transition
            tree.add_children([Melody((left_pitch,right_pitch),latent_variables=new_latent_variables)])
    return tree

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
        'harmony': [[0, 4, 7],'_',[0, 4, 7],'_',[2, 5, 7,11]],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

seq_2 = {
    'melody': [-5, '_', 5, '_', 4],
    'latent_info': {
        'harmony': [[2,5,7, 11],'_',[2,5,7, 11],'_',[0, 4, 7]],
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

melody_templates = [beginning,
                    second,
                    seq_1,
                    seq_2,
                    seq_1,
                    seq_2,
                    pre_cadence,
                    cadence]

tree_templates = [old_template_to_tree(x) for x in melody_templates]

if __name__ == '__main__':

    for template in tree_templates:
        print(template.children[0].latent_variables)
