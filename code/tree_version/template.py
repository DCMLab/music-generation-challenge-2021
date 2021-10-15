from melody import Melody


def old_template_to_tree(old_template):
    latent_variables = beginning['latent_info']
    tree = Melody('root',latent_variables=latent_variables)
    old_melody = old_template['melody']
    for i,x in enumerate(old_melody):
        if x=='_':
            left_pitch = old_melody[i-1]
            right_pitch = old_melody[i + 1]
            tree.add_children([Melody((left_pitch,right_pitch),latent_variables=latent_variables)])
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
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

seq_2 = {
    'melody': [-5, '_', 5, '_', 4],
    'latent_info': {
        'harmony': [2, 7, 11],
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
    print(second_tree.show())
    print('---')
    print(tree_templates[1].show())
