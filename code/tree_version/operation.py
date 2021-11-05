import copy
import random

from melody import Melody, Note
from pc_helpers import move_in_scale, scale_notes_between, harmony_notes_between


class Operation:
    def __init__(self, type_of_operation):
        self.type_of_operation = None

    @staticmethod
    def is_legal(melody: Melody):
        pass

    @staticmethod
    def add_children_by_pitch(melody: Melody, pitch: int, part, which_duration_to_steal):
        # if melody.part is head or tail overwrite which_duation_to_steal
        print('melody.part: ',melody.part)
        if melody.part == 'head':
            which_duration_to_steal=1
        elif melody.part == 'tail':
            which_duration_to_steal=0
        print('which_duration_to_steal: ', which_duration_to_steal)
        latent_variables = melody.transition[0].latent_variables
        transition = melody.transition
        which_note_to_give_duration = transition[which_duration_to_steal]
        halved_rhythm_cat = which_note_to_give_duration.rhythm_cat / 2
        which_note_to_give_duration.rhythm_cat = halved_rhythm_cat
        print('root_transition:',melody.get_root().transition)
        surface = (melody.get_root()).get_surface()
        location = melody.get_location_in_siblings()
        print('location: ',location)
        if which_duration_to_steal == 0:
            if location>0:
                print('stealing duration from left',location-1)
                surface[location-1].transition[1].rhythm_cat = halved_rhythm_cat

        if which_duration_to_steal == 1:
            if location < len(surface)-1:
                print('stealing duration from right',location+1)
                surface[location+1].transition[0].rhythm_cat = halved_rhythm_cat
        left_note, right_note = transition
        added_note = Note(pitch_cat=pitch, rhythm_cat=halved_rhythm_cat, latent_variables=latent_variables)
        child1 = Melody((left_note, added_note), part=part)
        child2 = Melody((added_note, right_note), part=part)
        melody.add_children([child1, child2])

class Repeat(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        left_duration,right_duration = melody.transition[0].rhythm_cat, melody.transition[1].rhythm_cat
        latent_variables = melody.transition[1].latent_variables
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            right_pitch != left_pitch,
            #left_duration >= 0.5,
            #right_duration >= 0.5,
            #right_pitch % 12 in latent_variables['harmony'],
            left_pitch % 12 in latent_variables['harmony']
        ])
        return False

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        Operation.add_children_by_pitch(melody, left_pitch, part=melody.part, which_duration_to_steal=0)

class Neighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        condition = all([
            left_pitch == right_pitch,
            left_pitch is not None,
            right_pitch is not None,
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        latent_variables = melody.transition[0].latent_variables
        scale = latent_variables['scale']
        register = left_pitch // 12
        neighbor_pitch = register * 12 + \
                         sorted([x for x in scale if x != left_pitch % 12],
                                key=lambda pitch: abs(pitch - left_pitch % 12))[
                             0]
        Operation.add_children_by_pitch(melody, neighbor_pitch, part=melody.part, which_duration_to_steal=0)


class Fill(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Fill')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        if {left_pitch, right_pitch}.intersection({'start', 'end'}):
            return False
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        latent_variables = melody.transition[0].latent_variables
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_variables['scale'])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            len(scale_notes_in_between) > 0,
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        midpoint = (right_pitch + left_pitch) / 2
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        latent_variables = melody.transition[0].latent_variables
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_variables['scale'])
        # print('scale_notes_in_between: ', scale_notes_in_between)
        harmony_notes_in_between = harmony_notes_between(low_pitch, high_pitch,
                                                         harmony=latent_variables['harmony'])
        # print('harmony_notes_between: ', harmony_notes_in_between)
        pitch_evaluation = lambda pitch: (1 / (1e-2 + abs(pitch - midpoint))) * (pitch % 12 in harmony_notes_in_between)
        # print(list(map(pitch_evaluation,scale_notes_between)))
        fill_pitch = sorted(scale_notes_in_between, key=pitch_evaluation)[-1]
        Operation.add_children_by_pitch(melody, fill_pitch, part=melody.part, which_duration_to_steal=0)


class RightNeighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='RightNeighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat

        if right_pitch == 'end' or left_pitch == 'start':
            return False
        latent_variables = melody.transition[0].latent_variables
        # print('melody.value: ', melody.transition[0].pitch_cat,melody.transition[1].pitch_cat)
        # print('harmony: ',latent_variables['harmony'])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            left_pitch != right_pitch,
            left_pitch % 12 in latent_variables['harmony']
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        latent_variables = melody.transition[0].latent_variables
        right_neighbor_pitch = move_in_scale(start_pitch=left_pitch, scale=latent_variables['scale'], step=-sign)
        Operation.add_children_by_pitch(melody, right_neighbor_pitch, part=melody.part, which_duration_to_steal=1)


class LeftNeighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='RightNeighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        if right_pitch == 'end' or left_pitch == 'start':
            return False
        latent_variables = melody.transition[1].latent_variables
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            left_pitch != right_pitch,
            right_pitch % 12 in latent_variables['harmony']
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        latent_variables = melody.transition[0].latent_variables
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        left_neighbor_pitch = move_in_scale(start_pitch=right_pitch, scale=latent_variables['scale'], step=sign)
        Operation.add_children_by_pitch(melody, left_neighbor_pitch, part=melody.part, which_duration_to_steal=0)
