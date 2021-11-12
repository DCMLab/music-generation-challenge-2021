import copy
import random

from melody import Melody, Note
from pc_helpers import move_in_scale, scale_notes_between, harmony_notes_between


class Operation:
    type_of_operation = 'None'

    @staticmethod
    def is_legal(melody: Melody):
        pass

    @staticmethod
    def exist_time_stealable(melody: Melody):
        if melody.part == 'tail':
            time_stealable_notes = [note for note in melody.transition[:1] if note.time_stealable]
        elif melody.part == 'body':
            time_stealable_notes = [note for note in melody.transition if note.time_stealable]
        elif melody.part == 'head':
            time_stealable_notes = [note for note in melody.transition[1:] if note.time_stealable]
        else:
            assert False
        not_resulting_32_notes = any([note.rhythm_cat > 0.25 for note in time_stealable_notes])
        return bool(time_stealable_notes) and not_resulting_32_notes

    @staticmethod
    def add_children_by_pitch(melody: Melody, pitch: int, part):
        # if melody.part is head or tail overwrite which_duation_to_steal
        print('melody.part: ', melody.part)
        time_stealable_notes = [note for note in melody.transition if note.time_stealable]
        assert time_stealable_notes
        durations = [note.rhythm_cat for note in melody.transition]
        which_duration_to_steal = durations.index(max(durations))
        if not melody.transition[which_duration_to_steal].time_stealable:
            which_duration_to_steal = 1 - which_duration_to_steal
        if melody.part == 'head':
            which_duration_to_steal = 1
        elif melody.part == 'tail':
            which_duration_to_steal = 0
        print('which_duration_to_steal: ', which_duration_to_steal)
        latent_variables = melody.transition[0].latent_variables
        transition = melody.transition
        which_note_to_give_duration = transition[which_duration_to_steal]
        halved_rhythm_cat = which_note_to_give_duration.rhythm_cat / 2
        which_note_to_give_duration.rhythm_cat = halved_rhythm_cat
        print('root_transition:', melody.get_root().transition)
        surface = (melody.get_root()).get_surface()
        location = melody.get_location_in_siblings()
        print('location: ', location)
        if which_duration_to_steal == 0:
            if location > 0:
                print('stealing duration from left', location - 1)
                surface[location - 1].transition[1].rhythm_cat = halved_rhythm_cat

        if which_duration_to_steal == 1:
            if location < len(surface) - 1:
                print('stealing duration from right', location + 1)
                surface[location + 1].transition[0].rhythm_cat = halved_rhythm_cat
        left_note, right_note = transition
        added_note = Note(pitch_cat=pitch, rhythm_cat=halved_rhythm_cat, latent_variables=latent_variables)
        child1 = Melody((left_note, added_note), part=part)
        child2 = Melody((added_note, right_note), part=part)
        melody.add_children([child1, child2])


class LeftRepeat(Operation):
    type_of_operation = 'LeftRepeat'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        left_duration, right_duration = melody.transition[0].rhythm_cat, melody.transition[1].rhythm_cat
        latent_variables = melody.transition[0].latent_variables
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            right_pitch != left_pitch,
            # left_duration >= 0.5,
            # right_duration >= 0.5,
            # right_pitch % 12 in latent_variables['harmony'],
            left_pitch % 12 in latent_variables['harmony'],
            left_duration > 0.5,
            right_duration > 0.5,

        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        Operation.add_children_by_pitch(melody, left_pitch, part=melody.part)


class RightRepeat(Operation):
    type_of_operation = 'RightRepeat'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        left_duration, right_duration = melody.transition[0].rhythm_cat, melody.transition[1].rhythm_cat
        latent_variables = melody.transition[1].latent_variables
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            right_pitch != left_pitch,
            # left_duration >= 0.5,
            # right_duration >= 0.5,
            # right_pitch % 12 in latent_variables['harmony'],
            right_pitch % 12 in latent_variables['harmony'],
            left_duration > 0.5,
            right_duration > 0.5,

        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        Operation.add_children_by_pitch(melody, right_pitch, part=melody.part)


class Neighbor(Operation):
    type_of_operation = 'Neighbor'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        surface_pitches = [note.pitch_cat for note in melody.get_root().surface_to_note_list(part='body')]
        register = left_pitch // 12
        neighbor_pitch = register * 12 + sorted([x for x in melody.transition[0].latent_variables['scale'] if x != left_pitch % 12],key=lambda pitch: abs(pitch - left_pitch % 12))[0]
        inserted_pitch_not_extreme_in_bar = min(surface_pitches) < neighbor_pitch < max(surface_pitches)
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch == right_pitch,
            left_pitch is not None,
            right_pitch is not None,
            inserted_pitch_not_extreme_in_bar,
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
        Operation.add_children_by_pitch(melody, neighbor_pitch, part=melody.part)


class Fill(Operation):
    type_of_operation = 'Fill'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        if {left_pitch, right_pitch}.intersection({'start', 'end'}):
            return False
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        latent_variables = melody.transition[0].latent_variables
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_variables['scale'])
        harmony_notes_in_between = harmony_notes_between(low_pitch, high_pitch,
                                                         harmony=latent_variables['harmony'])
        contain_harmony_notes = bool(harmony_notes_in_between)
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            len(scale_notes_in_between) > 0,
            contain_harmony_notes
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
        pitch_evaluation = lambda pitch: (1 / (1 + abs(pitch - midpoint))) + 10**(pitch % 12 in harmony_notes_in_between)
        # print(list(map(pitch_evaluation,scale_notes_between)))
        fill_pitch = sorted(scale_notes_in_between, key=pitch_evaluation)[-1]
        Operation.add_children_by_pitch(melody, fill_pitch, part=melody.part)


class RightNeighbor(Operation):
    type_of_operation = 'RightNeighbor'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat

        if right_pitch == 'end' or left_pitch == 'start':
            return False
        if right_pitch == left_pitch:
            return False
        # print('melody.value: ', melody.transition[0].pitch_cat,melody.transition[1].pitch_cat)
        # print('harmony: ',latent_variables['harmony'])
        surface_pitches = [note.pitch_cat for note in melody.get_root().surface_to_note_list(part='body')]
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        right_neighbor_pitch = move_in_scale(start_pitch=left_pitch,
                                             scale=melody.transition[0].latent_variables['scale'], step=-sign)
        inserted_pitch_not_extreme_in_bar = min(surface_pitches) < right_neighbor_pitch < max(surface_pitches)
        interval_size_not_big = abs(right_pitch - left_pitch) < 7
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            left_pitch % 12 in melody.transition[0].latent_variables['harmony'],
            inserted_pitch_not_extreme_in_bar or interval_size_not_big,
            right_pitch < left_pitch,
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        right_neighbor_pitch = move_in_scale(start_pitch=left_pitch,
                                             scale=melody.transition[0].latent_variables['scale'], step=-sign)
        Operation.add_children_by_pitch(melody, right_neighbor_pitch, part=melody.part)


class LeftNeighbor(Operation):
    type_of_operation = 'LeftNeighbor'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        if right_pitch == 'end' or left_pitch == 'start':
            return False
        if right_pitch == left_pitch:
            return False
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        left_neighbor_pitch = move_in_scale(start_pitch=right_pitch,
                                            scale=melody.transition[1].latent_variables['scale'], step=sign)
        surface_pitches = [note.pitch_cat for note in melody.get_root().surface_to_note_list(part='body')]
        inserted_pitch_not_extreme_in_bar = min(surface_pitches) < left_neighbor_pitch < max(surface_pitches)
        interval_size_not_big = abs(right_pitch-left_pitch)<5
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            right_pitch % 12 in melody.transition[1].latent_variables['harmony'],
            inserted_pitch_not_extreme_in_bar or interval_size_not_big

        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        left_neighbor_pitch = move_in_scale(start_pitch=right_pitch,
                                            scale=melody.transition[1].latent_variables['scale'], step=sign)
        print('scale: ', melody.transition[1].latent_variables['scale'])
        print('right_pitch: ', right_pitch)
        print('left_neighbor_pitch: ', left_neighbor_pitch)
        Operation.add_children_by_pitch(melody, left_neighbor_pitch, part=melody.part)
