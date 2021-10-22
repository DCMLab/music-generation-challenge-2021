from melody import Melody
from pc_helpers import move_in_scale, scale_notes_between, harmony_notes_between


class Operation:
    def __init__(self, type_of_operation):
        self.type_of_operation = None

    @staticmethod
    def is_legal(melody: Melody):
        pass

    @staticmethod
    def add_children_by_pitch(melody: Melody, pitch: int,part):
        left_pitch, right_pitch = melody.value

        child1 = Melody((left_pitch, pitch), latent_variables=melody.latent_variables,part=part)
        child2 = Melody((pitch, right_pitch), latent_variables=melody.latent_variables,part=part)
        melody.add_children([child1, child2])
    @staticmethod
    def perform(melody: Melody):
        pass


class Neighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.value
        condition = all([
            left_pitch == right_pitch,
            left_pitch is not None,
            right_pitch is not None,
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.value
        scale = melody.latent_variables['scale']
        register = left_pitch // 12
        neighbor_pitch = register * 12 + \
                         sorted([x for x in scale if x != left_pitch % 12],
                                key=lambda pitch: abs(pitch - left_pitch % 12))[
                             0]
        Operation.add_children_by_pitch(melody, neighbor_pitch,part=melody.part)


class Fill(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Fill')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.value
        if {left_pitch,right_pitch}.intersection({'start','end'}):
            return False
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=melody.latent_variables['scale'])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            len(scale_notes_in_between) > 0,
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.value
        midpoint = (right_pitch + left_pitch) / 2
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=melody.latent_variables['scale'])
        # print('scale_notes_in_between: ', scale_notes_in_between)
        harmony_notes_in_between = harmony_notes_between(low_pitch, high_pitch,
                                                         harmony=melody.latent_variables['harmony'][0])
        # print('harmony_notes_between: ', harmony_notes_in_between)
        pitch_evaluation = lambda pitch: (1 / (1e-2 + abs(pitch - midpoint))) * (pitch % 12 in harmony_notes_in_between)
        # print(list(map(pitch_evaluation,scale_notes_between)))
        fill_pitch = sorted(scale_notes_in_between, key=pitch_evaluation)[-1]
        Operation.add_children_by_pitch(melody, fill_pitch,part=melody.part)


class RightNeighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='RightNeighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.value

        if right_pitch == 'end' or left_pitch == 'start':
            return False
        print('melody.value: ', melody.value)
        print('harmony: ',melody.latent_variables['harmony'][0])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            left_pitch != right_pitch,
            left_pitch % 12 in melody.latent_variables['harmony'][0]
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.value
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        right_neighbor_pitch = move_in_scale(start_pitch=left_pitch, scale=melody.latent_variables['scale'], step=-sign)
        Operation.add_children_by_pitch(melody, right_neighbor_pitch,part=melody.part)


class LeftNeighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='RightNeighbor')

    @staticmethod
    def is_legal(melody: Melody):
        left_pitch, right_pitch = melody.value
        if right_pitch == 'end' or left_pitch == 'start':
            return False
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            left_pitch != right_pitch,
            right_pitch % 12 in melody.latent_variables['harmony'][1]
        ])
        return condition

    @staticmethod
    def perform(melody: Melody):
        left_pitch, right_pitch = melody.value
        sign = (right_pitch - left_pitch) / abs(right_pitch - left_pitch)
        left_neighbor_pitch = move_in_scale(start_pitch=right_pitch, scale=melody.latent_variables['scale'], step=sign)
        Operation.add_children_by_pitch(melody, left_neighbor_pitch,part=melody.part)
