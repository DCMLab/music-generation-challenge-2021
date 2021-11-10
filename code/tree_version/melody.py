import music21 as m21


class Note:
    def __init__(self, pitch_cat, rhythm_cat, latent_variables: dict,time_stealable=True):
        self.pitch_cat = pitch_cat
        self.rhythm_cat = rhythm_cat
        self.latent_variables = latent_variables
        self.time_stealable = time_stealable


class Tree:
    def __init__(self):
        self.children = []
        self.parent = None
        self.memory = None

    def add_children(self, children):
        self.children += children
        for child in children:
            child.parent = self

    def get_depth(self):
        if not self.children:
            return 0
        return max([child.get_depth() for child in self.children]) + 1

    def get_root(self):
        if not self.parent:
            return self
        else:
            return self.parent.get_root()

    def get_dist_to_root(self):
        if not self.parent:
            return 0
        return 1 + self.parent.get_dist_to_root()

    def get_subtrees_at_depth(self, depth):
        children = [self]
        while depth > 0:
            children = sum([child.children for child in children], [])
            depth = depth - 1
        return children

    def get_surface_at_depth(self, depth):
        surface = [self]
        while depth > 0:
            sub_surface = []
            for tree in surface:
                if not tree.children:
                    sub_surface.append(tree)
                else:
                    sub_surface.extend(tree.children)
            surface = sub_surface
            depth = depth - 1
        return surface

    def get_surface(self):
        max_depth = self.get_depth()
        return self.get_surface_at_depth(max_depth)

    def get_location_in_siblings(self):
        surface = (self.get_root()).get_surface()
        location = [i for i, x in enumerate(surface) if x is self][0]
        return location

    def show(self):
        depth = self.get_depth()
        for i in range(depth + 1):
            subtrees = self.get_surface_at_depth(i)
            print([(x.symbol_cat,x.rhythm_cat,x.latent_variables['harmony']) for x in subtrees])


class Melody(Tree):
    def __init__(self, transition=(Note('start', 'start', {}), Note('end', 'end', {})), part='body',no_tail = False):
        super().__init__()
        self.surface = None
        self.transition = transition
        self.part = part  # head, body, or tail region of root
        self.no_tail = no_tail

    def show(self):
        depth = self.get_depth()
        for i in range(depth + 1):
            subtrees = self.get_surface_at_depth(i)
            print([tuple(map(lambda _: _.__dict__, x.transition)) for x in subtrees])

    def surface_to_note_list(self):
        surface = self.get_surface()
        head_region = [x for x in surface if x.part == 'head']
        body_region = [x for x in surface if x.part == 'body']
        tail_region = [x for x in surface if x.part == 'tail']

        head_region_note_list = [melody.transition[1] for melody in head_region[:-1]]
        body_region_note_list = [body_region[0].transition[0], body_region[0].transition[1]] + [melody.transition[1] for
                                                                                                melody in
                                                                                                body_region[1:]]
        tail_region_note_list = [melody.transition[1] for melody in tail_region[:-1]]
        note_list = head_region_note_list + body_region_note_list + tail_region_note_list
        #print('*****')
        #print('pitch_dur_list: ', [(note.pitch_cat, note.rhythm_cat) for note in note_list])

        # note_list = [surface[0].transition[0], surface[0].transition[1]] + [melody.transition[1] for melody in surface[1:]]

        return note_list

    def surface_to_stream(self):
        note_list = self.surface_to_note_list()
        measure = m21.stream.Measure()
        for note in note_list:
            m21_note = m21.note.Note(pitch=60 + note.pitch_cat, quarterLength=note.rhythm_cat)
            measure.append(m21_note)
        return measure

    def get_total_duration(self):
        note_list = self.surface_to_note_list()
        durations = [note.rhythm_cat for note in note_list]
        total_duration = sum(durations)
        return total_duration


scale = [0, 2, 4, 5, 7, 9, 11]
latent_variables = {'harmony': [0, 4, 7], 'scale': scale}

seq_1 = Melody()
seq_1.add_children([Melody(
    transition=(Note(-5, 1.0, latent_variables=latent_variables), Note(7, 1.0, latent_variables=latent_variables))),
                    Melody(transition=(Note(7, 1.0, latent_variables=latent_variables),
                                       Note(5, 1.0, latent_variables={'harmony': [2, 5, 7, 11], 'scale': scale})))
                    ])

if __name__ == '__main__':
    seq_1.show()
    seq_1.surface_to_stream().show()
    # print('pitch_rhythm_surface: ', [x for x in tree.get_surface()])
