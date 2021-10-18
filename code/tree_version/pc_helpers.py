
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
    scale_notes_in_between = [x for x in chromatic_notes_between if x % 12 in scale]
    return scale_notes_in_between


def harmony_notes_between(low_pitch, high_pitch, harmony):
    return scale_notes_between(low_pitch, high_pitch, harmony)

def interval_list_to_pitch_list(interval_list: list[(int, int)]) -> list[int]:
    pitch_list = []
    for i, pair in enumerate(interval_list):
        if i == 0:
            pitch_list.extend(pair)
        else:
            pitch_list.append(pair[1])
    return pitch_list