import copy
from melody import Melody, Note




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

seq_1 = Melody(no_tail=True)
seq_1.add_children([Melody(
    transition=(Note(-5, 1.0, latent_variables=latent_variables), Note(7, 1.0, latent_variables=latent_variables))),
    Melody(transition=(Note(7, 1.0, latent_variables=latent_variables),
                       Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale})))
])

seq_2 = Melody(no_tail=True)
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
                        Note(0, 1.0, latent_variables={'harmony': [0,4,7], 'scale': scale},time_stealable=False))),
     Melody(transition=(Note(0, 1.0, latent_variables={'harmony': [0,4,7], 'scale': scale},time_stealable=False),
                        Note(-12, 1.0, latent_variables={'harmony': [0,4,7], 'scale': scale},time_stealable=False)))
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
                        Note(5, 1.0, latent_variables={'harmony': [0, 5, 9], 'scale': [0, 2, 4, 5, 7, 9, 10]})))
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
                        Note(7, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale}))),
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
                        Note(0, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale},time_stealable=False))),
     Melody(transition=(Note(0, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale},time_stealable=False),
                        Note(-12, 1.0, latent_variables={'harmony': [0, 4, 7], 'scale': scale},time_stealable=False)))
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
                        'h', 'i', 'h', 'x',
                        'j', 'j', 'k', 'l']