import music21 as m21

class Converter:
    @staticmethod
    def melody_list_to_m21_measure(melody_list: list) -> m21.stream.Measure:
        measure = m21.stream.Measure()
        pitches = [x for x in melody_list if x != '_']
        for pitch in pitches:
            measure.append(m21.note.Note(pitch=60+pitch))
        return measure

