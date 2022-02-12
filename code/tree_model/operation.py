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
        # only steal time from left
        #time_stealable_notes = [note for note in melody.transition[:1] if note.time_stealable]
        if time_stealable_notes:
            not_resulting_32_notes = any([note.rhythm_cat > 0.25 for note in time_stealable_notes])
            return not_resulting_32_notes
        else:
            return False

    @classmethod
    def add_children_by_pitch(cls,melody: Melody, pitch: int, part):
        # if melody.part is head or tail overwrite which_duation_to_steal

        durations = [note.rhythm_cat for note in melody.transition if note.time_stealable]
        max_duration = max(durations)

        if max_duration > 0.5:
            which_duration_to_steal = [i for i, x in enumerate(durations) if x == max_duration][0]
        else:
            which_duration_to_steal = [i for i, x in enumerate(durations) if x == max_duration][-1]
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

        surface = melody.get_root().get_surface()
        location = melody.get_location_in_siblings()

        if which_duration_to_steal == 0:
            if location > 0:
                print('stealing duration from left', location - 1)
                surface[location - 1].transition[1].rhythm_cat = halved_rhythm_cat

        if which_duration_to_steal == 1:
            if location < len(surface) - 1:
                print('stealing duration from right', location + 1)
                surface[location + 1].transition[0].rhythm_cat = halved_rhythm_cat
        left_note, right_note = transition
        if pitch % 12 == 7:
            if {4, 8, 11}.issubset(set(left_note.latent_variables['harmony'])) or (
                    right_note.pitch_cat % 12 == 9 and left_note.pitch_cat % 12 == 7):
                new_pitch = pitch + 1
            else:
                new_pitch = pitch
        else:
            new_pitch = pitch
        added_note = Note(pitch_cat=new_pitch, rhythm_cat=halved_rhythm_cat, latent_variables=latent_variables,source_operation=cls.type_of_operation)
        child1 = Melody((left_note, added_note), part=part)
        child2 = Melody((added_note, right_note), part=part)
        melody.add_children([child1, child2])


class Initialize(Operation):
    type_of_operation = 'Initialize'

    @staticmethod
    def is_legal(melody: Melody):
        return

    @staticmethod
    def perform(melody: Melody):
        pass


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
        return False

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
        RightRepeat.add_children_by_pitch(melody, right_pitch, part=melody.part)


class Neighbor(Operation):
    type_of_operation = 'Neighbor'

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        surface_pitches = [note.pitch_cat for note in melody.get_root().surface_to_note_list(part='body')]
        register = left_pitch // 12
        neighbor_pitch = register * 12 + \
                         sorted([x for x in melody.transition[0].latent_variables['scale'] if x != left_pitch % 12],
                                key=lambda pitch: abs(pitch - left_pitch % 12))[0]
        inserted_pitch_not_extreme_in_bar = min(surface_pitches) < neighbor_pitch < max(surface_pitches)
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch == right_pitch,
            left_pitch is not None,
            right_pitch is not None,
            inserted_pitch_not_extreme_in_bar,
        ])
        return False

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
        Neighbor.add_children_by_pitch(melody, neighbor_pitch, part=melody.part)


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
            len(scale_notes_in_between) > 0 or melody.transition[0].rhythm_cat > 0.25,
            contain_harmony_notes or len(scale_notes_in_between) == 1
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        midpoint = (right_pitch + left_pitch) / 2
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        latent_variables = melody.transition[0].latent_variables
        if {4,8,11}.issubset(latent_variables['harmony']):
            scale = sorted(latent_variables['scale']+[8])
        else:
            scale = latent_variables['scale']
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=scale)
        # print('scale_notes_in_between: ', scale_notes_in_between)
        pitch_evaluation = lambda pitch: (1 / (1 + abs(pitch - midpoint))) + 10 ** (
                    pitch % 12 in latent_variables['harmony'])
        # print(list(map(pitch_evaluation,scale_notes_between)))
        fill_pitch = sorted(scale_notes_in_between, key=pitch_evaluation)[-1]
        Fill.add_children_by_pitch(melody, fill_pitch, part=melody.part)


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
        interval_size_not_big = abs(right_pitch - left_pitch) <= 7
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            #left_pitch % 12 in melody.transition[0].latent_variables['harmony'],
            right_pitch % 12 in melody.transition[1].latent_variables['harmony'],
            inserted_pitch_not_extreme_in_bar or melody.transition[0].pitch_cat > 0.5,
            right_pitch <= left_pitch,
            interval_size_not_big,
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        right_neighbor_pitch = move_in_scale(start_pitch=left_pitch,
                                             scale=melody.transition[0].latent_variables['scale'], step=-sign)
        RightNeighbor.add_children_by_pitch(melody, right_neighbor_pitch, part=melody.part)


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
        interval_size_not_big = abs(right_pitch - left_pitch) < 7
        condition = all([
            Operation.exist_time_stealable(melody),
            left_pitch is not None,
            right_pitch is not None,
            right_pitch % 12 != 8,  # avoid approach raised leading tone from below
            # left_pitch % 12 in melody.transition[0].latent_variables['harmony'],
            right_pitch % 12 in melody.transition[1].latent_variables['harmony'],
            # inserted_pitch_not_extreme_in_bar, #or melody.transition[0].pitch_cat>0.5,
            interval_size_not_big,

        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.transition[0].pitch_cat, melody.transition[1].pitch_cat
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        left_neighbor_pitch = move_in_scale(start_pitch=right_pitch,
                                            scale=melody.transition[1].latent_variables['scale'], step=sign)
        LeftNeighbor.add_children_by_pitch(melody, left_neighbor_pitch, part=melody.part)
