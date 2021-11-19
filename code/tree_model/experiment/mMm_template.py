import copy

from ..melody import Melody, Note


scale = [0, 2, 4, 5, 7,9,11]
M6scale = [1,2,4,6,8,9,11]
latent_variables = {'harmony': [0, 4, 9], 'scale': scale}
mV_latent_variables = {'harmony': [2, 4, 8, 11], 'scale': scale}
M_latent_variables = {'harmony': [0, 4, 7], 'scale': scale}
MV_latent_variables = {'harmony': [2, 5, 7,11], 'scale': scale}
beginning = Melody(max_elaboration=4)
beginning.add_children(
    [Melody(transition=(Note(12+4, 1.0, latent_variables=latent_variables), Note(12, 1.0, latent_variables=mV_latent_variables))),
     Melody(transition=(Note(12, 1.0, latent_variables=mV_latent_variables), Note(9, 1.0, latent_variables=latent_variables)))
     ])

second = Melody(max_elaboration=4,no_tail=True)
second.add_children(
    [Melody(transition=(Note(4, 1.0, latent_variables={'harmony': [1, 4, 9], 'scale': M6scale}), Note(12+1, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': M6scale}))),
     Melody(transition=(Note(12+1, 1.0, latent_variables={'harmony': [1, 4, 9], 'scale': M6scale}), Note(9, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': M6scale})))
     ])

first_cadence = Melody(max_elaboration=3,no_tail=True)
first_cadence.add_children([
    Melody(transition=(Note(4, 1.0, latent_variables=mV_latent_variables), Note(9, 1.0, latent_variables=latent_variables,time_stealable=False))),
    Melody(transition=(Note(9, 1.0, latent_variables=latent_variables,time_stealable=False), Note(-3, 1.0, latent_variables=latent_variables,time_stealable=False)))
     ])

seq1 = Melody(max_elaboration=6)
seq1.add_children([
    Melody(transition=(Note(12+7, 1.0, latent_variables=M_latent_variables), Note(12+12, 1.0, latent_variables=M_latent_variables))),
    Melody(transition=(Note(12+12, 1.0, latent_variables=M_latent_variables), Note(12+4, 1.0, latent_variables=M_latent_variables)))
])

seq2 = Melody(max_elaboration=6,no_tail=True)
seq2.add_children([
    Melody(transition=(Note(12+4, 1.0, latent_variables=M_latent_variables), Note(12, 1.0, latent_variables=M_latent_variables))),
    Melody(transition=(Note(12, 1.0, latent_variables=M_latent_variables), Note(9, 1.0, latent_variables=M_latent_variables)))
])

second_cadence = Melody(max_elaboration=3,no_tail=True)
second_cadence.add_children([
    Melody(transition=(Note(12+4, 1.0, latent_variables=MV_latent_variables), Note(12, 1.0, latent_variables=M_latent_variables,time_stealable=False))),
    Melody(transition=(Note(12, 1.0, latent_variables=M_latent_variables,time_stealable=False), Note(0, 1.0, latent_variables=M_latent_variables,time_stealable=False)))
     ])

seq3 = Melody(max_elaboration=6,no_tail=True)
seq3.add_children([
    Melody(transition=(Note(12+4, 0.5, latent_variables=latent_variables), Note(9, 0.5, latent_variables=latent_variables))),
    Melody(transition=(Note(9, 0.5, latent_variables=latent_variables), Note(12+4, 0.5, latent_variables=latent_variables))),
    Melody(transition=(Note(12+4, 0.5, latent_variables=latent_variables), Note(9, 0.5, latent_variables=latent_variables))),
    Melody(transition=(Note(9, 0.5, latent_variables=latent_variables), Note(12+4, 0.5, latent_variables=latent_variables))),
    Melody(transition=(Note(12+4, 0.5, latent_variables=latent_variables), Note(9, 0.5, latent_variables=latent_variables))),
])

seq4 = Melody(max_elaboration=6,no_tail=True)
seq4.add_children([
    Melody(transition=(Note(12+2, 0.5, latent_variables=MV_latent_variables), Note(7, 0.5, latent_variables=MV_latent_variables))),
    Melody(transition=(Note(7, 0.5, latent_variables=MV_latent_variables), Note(12+2, 0.5, latent_variables=MV_latent_variables))),
    Melody(transition=(Note(12+2, 0.5, latent_variables=MV_latent_variables), Note(7, 0.5, latent_variables=MV_latent_variables))),
    Melody(transition=(Note(7, 0.5, latent_variables=MV_latent_variables), Note(12+2, 0.5, latent_variables=MV_latent_variables))),
    Melody(transition=(Note(12+2, 0.5, latent_variables=MV_latent_variables), Note(7, 0.5, latent_variables=MV_latent_variables))),
])

precadence = Melody(max_elaboration=6)
precadence.add_children([
    Melody(transition=(Note(12+8, 1.0, latent_variables=MV_latent_variables), Note(12+11, 1.0, latent_variables=MV_latent_variables))),
    Melody(transition=(Note(12+11, 1.0, latent_variables=MV_latent_variables), Note(12+2, 1.0, latent_variables=MV_latent_variables)))
])

third_candence = Melody(max_elaboration=5,no_tail=True)
third_candence.add_children([
    Melody(transition=(Note(11, 1.0, latent_variables=MV_latent_variables), Note(9, 1.0, latent_variables=M_latent_variables,time_stealable=False))),
    Melody(transition=(Note(9, 1.0, latent_variables=M_latent_variables,time_stealable=False), Note(-3, 1.0, latent_variables=M_latent_variables,time_stealable=False)))
     ])

mMm_template = [
    beginning,
    second,
    copy.deepcopy(beginning),
    first_cadence,
    #
    seq1,
    seq2,
    copy.deepcopy(seq1),
    second_cadence,
    #
    seq3,
    seq4,
    precadence,
    third_candence
]

mMm_similarity = [
    'a','b','a','d',
    'e','f','e','g',
    'h','h','i','j'
]