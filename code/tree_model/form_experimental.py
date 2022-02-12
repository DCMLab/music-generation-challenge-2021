from form import Form
scale = [0, 2, 4, 5, 7, 9, 11]
i_latent_variables = {'harmony': [0, 4, 7], 'scale': scale}
V_latent_variables = {'harmony': [2, 7, 11], 'scale': scale}

def build_self_similar_phrase():
    phrase = Form(rhythm_cat=32, symbol_cat='a', max_elaboration=20, latent_variables=i_latent_variables)
    phrase.add_children([
        Form(rhythm_cat=16, symbol_cat='a', max_elaboration=10, latent_variables=i_latent_variables),
        Form(rhythm_cat=16, symbol_cat='a', max_elaboration=10, latent_variables=i_latent_variables),
    ])
    for child in phrase.children:
        child.add_children([
            Form(rhythm_cat=8, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables),
            Form(rhythm_cat=8, symbol_cat='a', max_elaboration=5, latent_variables=i_latent_variables),
        ])

    for _child in phrase.children:
        for child in _child.children:
            child.add_children([
                Form(rhythm_cat=4, symbol_cat='a', max_elaboration=3, latent_variables=i_latent_variables),
                Form(rhythm_cat=4, symbol_cat='a', max_elaboration=3, latent_variables=i_latent_variables),
            ])
    return phrase

class FormElaboration:
    pass

class FormOperation:


def func():
    phrase = build_self_similar_phrase()
    sim_temp = phrase.to_similarity_template()
    melody_temp = phrase.to_melody_templates()
    print(sim_temp)
    melody_temp[0].show()


if __name__ == '__main__':
    func()
