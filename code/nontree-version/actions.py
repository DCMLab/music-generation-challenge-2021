import copy


def left_and_right_pitch(melody, slot):
    left_pitches = [x for i, x in enumerate(melody) if i < slot and x != '_']
    right_pitches = [x for i, x in enumerate(melody) if i > slot and x != '_']
    if len(left_pitches) > 0:
        left_pitch = left_pitches[-1]
    else:
        left_pitch = None
    if len(right_pitches) > 0:
        right_pitch = right_pitches[0]
    else:
        right_pitch = None
    # print('left_pitch, right_pitch: ', left_pitch, right_pitch)
    return left_pitch, right_pitch


def move_in_scale(start_pitch, scale, step):
    current_pitch = start_pitch
    sign = int(step / abs(step))
    while step != 0:
        current_pitch = current_pitch + sign
        if current_pitch % 12 in scale:
            step = step - sign
    return current_pitch


def scale_notes_between(low_pitch, high_pitch, scale):
    chromatic_notes_between = list(range(low_pitch + 1, high_pitch))
    scale_notes_in_between = [x for x in chromatic_notes_between if x%12 in scale]
    return scale_notes_in_between


def harmony_notes_between(low_pitch,high_pitch, harmony):
    return scale_notes_between(low_pitch,high_pitch, harmony)


class Operation:
    def __init__(self, type_of_operation):
        self.type_of_operation = None

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        pass

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        pass


class Neighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        """works when left pitch = right pitch, both present"""
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        condition = all([
            left_pitch == right_pitch,
            left_pitch is not None,
            right_pitch is not None,
        ])
        return condition

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        scale = latent_info['scale']
        register = left_pitch // 12
        neighbor_pitch = register*12 + \
                         sorted([x for x in scale if x != left_pitch%12], key=lambda pitch: abs(pitch - left_pitch % 12))[
                             0]
        new_melody = copy.deepcopy(melody)
        new_melody[slot:slot + 1] = '_', neighbor_pitch, '_'
        return new_melody


class Fill(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        """works when left pitch and right pitch both present, and there is a scale note in between"""
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_info['scale'])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            len(scale_notes_in_between) > 0,
        ])
        return condition

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        midpoint = (right_pitch + left_pitch) / 2
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_info['scale'])
        #print('scale_notes_in_between: ', scale_notes_in_between)
        harmony_notes_in_between = harmony_notes_between(low_pitch, high_pitch, harmony=latent_info['harmony'])
        #print('harmony_notes_between: ', harmony_notes_in_between)
        pitch_evaluation = lambda pitch: (1 / (1e-2 + abs(pitch - midpoint))) * (pitch % 12 in harmony_notes_in_between)
        # print(list(map(pitch_evaluation,scale_notes_between)))
        fill_pitch = sorted(scale_notes_in_between, key=pitch_evaluation)[-1]
        new_melody = copy.deepcopy(melody)
        new_melody[slot:slot + 1] = '_', fill_pitch, '_'
        return new_melody


class Escape(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        """works when left pitch and right pitch are one scale step apart"""
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_info['scale'])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            len(scale_notes_in_between) == 0,
            left_pitch!=right_pitch,
            right_pitch % 12 in latent_info['harmony']
        ])
        return condition

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        sign = (right_pitch-left_pitch)/abs(right_pitch-left_pitch)
        escape_pitch = move_in_scale(start_pitch=left_pitch, scale=latent_info['scale'], step=-sign)
        new_melody = copy.deepcopy(melody)
        new_melody[slot:slot + 1] = '_', escape_pitch, '_'
        return new_melody

class Appoggiatura(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        """works when left pitch and right pitch are different"""
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        low_pitch, high_pitch = sorted([left_pitch, right_pitch])
        scale_notes_in_between = scale_notes_between(low_pitch, high_pitch, scale=latent_info['scale'])
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            left_pitch != right_pitch,
            right_pitch%12 in latent_info['harmony']
        ])
        return condition

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        sign = (right_pitch-left_pitch)/abs(right_pitch-left_pitch)
        escape_pitch = move_in_scale(start_pitch=right_pitch, scale=latent_info['scale'], step=sign)
        new_melody = copy.deepcopy(melody)
        new_melody[slot:slot + 1] = '_', escape_pitch, '_'
        return new_melody
