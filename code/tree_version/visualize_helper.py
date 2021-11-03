import music21 as m21

class Converter:
    @staticmethod
    def melody_list_to_m21_measure(melody_list: list) -> m21.stream.Measure:
        measure = m21.stream.Measure()
        pitches = [x for x in melody_list if x != '_']
        for pitch in pitches:
            note = m21.note.Note(pitch=60+pitch)
            measure.append(note)
        return measure

    @staticmethod
    def melody_duration_list_to_m21_measure(melody_duration_list: list) -> m21.stream.Measure:
        measure = m21.stream.Measure()
        pitch_durations = [x for x in melody_duration_list if x[0] != '_' and x[1] != '_']
        for pitch_duration in pitch_durations:
            note = m21.note.Note(pitch=60 + pitch_duration[0])
            note.duration = pitch_duration[1]
            measure.append(note)
        return measure


