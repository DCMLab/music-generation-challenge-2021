from melody import Melody,Note


def move_in_scale(start_pitch, scale, step):
    current_pitch = start_pitch
    sign = int(step / abs(step))
    while step != 0:
        current_pitch = current_pitch + sign
        if current_pitch % 12 in scale:
            step = step - sign
    return current_pitch


def scale_notes_between(low_pitch, high_pitch, scale):
    assert isinstance(low_pitch,int) and isinstance(high_pitch,int) and isinstance(scale,list),(low_pitch,high_pitch,scale)
    chromatic_notes_between = list(range(low_pitch + 1, high_pitch))
    scale_notes_in_between = [x for x in chromatic_notes_between if x % 12 in scale]
    return scale_notes_in_between


def harmony_notes_between(low_pitch, high_pitch, harmony):
    return scale_notes_between(low_pitch, high_pitch, harmony)


def interval_list_to_pitch_list(interval_list: list[(int, int)]) -> list[int]:
    pitch_list = []
    for i, pair in enumerate(interval_list):
        if i == 0:
            if pair[0] == 'start':
                pitch_list.append(pair[1])
            else:
                pitch_list.extend(pair)
        else:
            if pair[1] == 'end':
                pass
            else:
                pitch_list.append(pair[1])
    return pitch_list


def melody_surface_to_pitch_list(surface: list[Melody]) -> list[int]:
    head_region = [x.value for x in surface if x.part == 'head']
    body_region = [x.value for x in surface if x.part == 'body']
    tail_region = [x.value for x in surface if x.part == 'tail']
    head_region_pitch_list = [interval[1] for interval in head_region[:-1]]
    body_region_pitch_list = [body_region[0][0]] + [interval[1] for interval in body_region]
    tail_region_pitch_list = [interval[1] for interval in tail_region[:-1]]

    pitch_list = head_region_pitch_list + body_region_pitch_list + tail_region_pitch_list
    return pitch_list



