import copy

from melody import Melody, Note


scale = [0, 2, 4, 5, 7,9,11]
latent_variables = {'harmony': [0, 4, 9], 'scale': scale}


beginning = Melody(max_elaboration=0)
beginning.add_children(
    [Melody(transition=(Note(12, 1.0, latent_variables=latent_variables), Note(11, 1.0, latent_variables=latent_variables))),
     Melody(transition=(Note(11 + 4, 1.0, latent_variables=latent_variables), Note(9, 1.0, latent_variables=latent_variables)))
     ])

second = Melody(max_elaboration=5,no_tail=True)
second.add_children(
    [Melody(transition=(Note(4, 1.0, latent_variables=latent_variables), Note(11, 1.0, latent_variables=latent_variables))),
     Melody(transition=(Note(11, 1.0, latent_variables=latent_variables), Note(9, 1.0, latent_variables=latent_variables)))
     ])

third = Melody(max_elaboration=5)
third.add_children(
    [Melody(transition=(Note(12+5, 1.0, latent_variables=latent_variables), Note(12+2, 1.0, latent_variables=latent_variables))),
     Melody(transition=(Note(12+2, 1.0, latent_variables=latent_variables), Note(12, 1.0, latent_variables=latent_variables)))
     ])

first_cadence = Melody(max_elaboration=5,no_tail=True)
first_cadence.add_children(
    [
     Melody(transition=(Note(8, 1.0, latent_variables={'harmony': [2, 4, 8,11], 'scale': scale}), Note(9, 2.0, latent_variables=latent_variables,time_stealable=False)))
     ])


seq_1 = Melody(max_elaboration=5)
seq_1.add_children([
    Melody(transition=(Note(9, 1.0, latent_variables=latent_variables), Note(12+4, 1.0, latent_variables=latent_variables))),
    Melody(transition=(Note(12+4, 1.0, latent_variables=latent_variables), Note(11, 1.0, latent_variables=latent_variables)))
])

seq_2 = Melody(max_elaboration=5)
seq_2.add_children([
    Melody(transition=(Note(7, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale}), Note(12+2, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale}))),
    Melody(transition=(Note(12+2, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale}), Note(4, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale})))
])

seq_3 = Melody(max_elaboration=6)
seq_3.add_children([
    Melody(transition=(Note(9, 1.0, latent_variables=latent_variables), Note(12+9, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale}))),
    Melody(transition=(Note(12+9, 1.0, latent_variables=latent_variables), Note(12+5, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale})))
])

seq_4 = Melody(max_elaboration=7)
seq_4.add_children([
    Melody(transition=(Note(12+2, 1.0, latent_variables=latent_variables), Note(12+7, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale}))),
    Melody(transition=(Note(12+7, 1.0, latent_variables=latent_variables), Note(12+4, 1.0, latent_variables={'harmony': [2,4, 7,11], 'scale': scale})))
])

minor_template = [
    beginning,
    second,
    third,
    first_cadence,
    seq_1,
    seq_2,
    copy.deepcopy(seq_1),
    copy.deepcopy(seq_2),
    seq_3,
    seq_4,
    copy.deepcopy(third),
    copy.deepcopy(first_cadence),
]

minor_similarity = [
    'a','b','c','d',
    'e','e','e','e',
    'f','f','c','d',
]