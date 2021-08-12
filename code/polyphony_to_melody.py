import random

import music21 as m21
import numpy as np
import music21.stream

major_key = 0
scale = np.array([0, 2, 4, 5, 7, 9, 11], dtype=int) + major_key


def diatonic_transpose(midi_number, scale, steps):
    octave = midi_number // 12
    degree = np.argwhere(scale == midi_number % 12)[0][0]
    new_midi = 12 * octave + scale[(degree + steps) % 7] + 12 * ((degree + steps) // 7)
    print(new_midi)
    return new_midi


class ChordHorizontalize:
    @staticmethod
    def arp(pitches, index_sequence, **kwargs):
        new_pitches = [pitches[x % len(pitches)] for x in index_sequence]
        return new_pitches

    @staticmethod
    def linear_prog(pitches, scale=None, start_end_indexes=(0, -1), **kwargs):
        starting_pitch = pitches[start_end_indexes[0]]
        end_pitch = pitches[start_end_indexes[1]]
        if len(pitches) == 2:
            # base case (one segment), works if the two pitches are 2 or 4 steps apart
            start_midi = starting_pitch.midi
            end_midi = end_pitch.midi
            start_degree = np.argwhere((start_midi - scale) % 12 == 0).reshape(-1)[0]
            print(start_degree)
            local_midi_window = np.concatenate([start_midi - scale[start_degree] + x + scale for x in [-12, 0, 12]])
            new_midis = local_midi_window[(start_midi <= local_midi_window) * (local_midi_window <= end_midi)]
            new_pitches = [m21.pitch.Pitch(midi=midi) for midi in new_midis]
        else:
            # multiple segments case
            new_pitches = ChordHorizontalize.linear_prog([starting_pitch, end_pitch], scale, start_end_indexes)

        return new_pitches

    @staticmethod
    def neighbor(pitches,scale=None):
        first_note = m21.note.Note(pitches[min(2 , len(pitches)-1)])
        first_note.duration.quarterLength = 0.25
        second_note = m21.note.Note(diatonic_transpose(first_note.pitch.midi, scale=scale, steps=2))
        second_note.duration.quarterLength = 0.25
        third_note = m21.note.Note(diatonic_transpose(first_note.pitch.midi, scale=scale, steps=1))
        third_note.duration.quarterLength = 0.25
        fourth_note = m21.note.Note(first_note.pitch)
        fourth_note.duration.quarterLength = 0.25
        pitches = [x.pitch for x in [first_note,second_note,third_note,fourth_note]]
        return pitches

    @staticmethod
    def neighbor_2(pitches, scale=None):
        first_note = m21.note.Note(pitches[min(2, len(pitches) - 1)])
        first_note.duration.quarterLength = 0.25
        second_note = m21.note.Note(diatonic_transpose(first_note.pitch.midi, scale=scale, steps=-2))
        second_note.duration.quarterLength = 0.25
        third_note = m21.note.Note(diatonic_transpose(first_note.pitch.midi, scale=scale, steps=-1))
        third_note.duration.quarterLength = 0.25
        fourth_note = m21.note.Note(diatonic_transpose(first_note.pitch.midi, scale=scale, steps=0))
        fourth_note.duration.quarterLength = 0.25
        pitches = [x.pitch for x in [first_note, second_note, third_note, fourth_note]]
        return pitches




class BarHorizontalize:
    @staticmethod
    def pattern1(chords):
        guide_voices = [1, -1]
        new_measure = m21.stream.Measure()
        all_pitches = [chord.pitches for chord in chords]
        for i, pitches in enumerate(all_pitches):
            first_note = m21.note.Note(pitches[min(len(pitches)-1,2)])
            first_note.duration.quarterLength = 0.25
            second_note = m21.note.Note(diatonic_transpose(first_note.pitch.midi, scale=scale, steps=1))
            second_note.duration.quarterLength = 0.25
            third_note = m21.note.Note(first_note.pitch)
            third_note.duration.quarterLength = 0.25
            if i < len(all_pitches) - 1:
                next_first_pitch = all_pitches[i + 1][min(2,len(all_pitches[i + 1])-1)]
                print(next_first_pitch.midi)
                pitch = m21.pitch.Pitch(diatonic_transpose(next_first_pitch.midi, scale=scale, steps=-1))
                print(pitch.midi)
                fourth_note = m21.note.Note(pitch)
                fourth_note.duration.quarterLength = 0.25

            else:
                pitch = diatonic_transpose(third_note.pitch.midi, scale=scale, steps=-1)
                fourth_note = m21.note.Note(pitch)
                fourth_note.duration.quarterLength = 0.25

            new_measure.append(first_note)
            new_measure.append(second_note)
            new_measure.append(third_note)
            new_measure.append(fourth_note)
        return new_measure
    @staticmethod
    def pattern2(chords):
        guide_voices = [0, 1, 2]
        new_measure = m21.stream.Measure()
        all_pitches = [chord.pitches for chord in chords]
        for i, pitches in enumerate(all_pitches):
            if i in [0]:
                new_pitches = ChordHorizontalize.arp(pitches, [0, 1, 2, 1])
            if i in [1,2]:
                new_pitches = ChordHorizontalize.arp(pitches, [2, 0, 1, 0])
            for pitch in new_pitches:
                note = m21.note.Note(pitch=pitch)
                note.duration.quarterLength = 0.25
                new_measure.append(note)
        return new_measure

    @staticmethod
    def pattern3(chords):
        new_measure = m21.stream.Measure()
        all_pitches = [chord.pitches for chord in chords]
        for i, pitches in enumerate(all_pitches):
            if i in [0,1]:
                new_pitches = ChordHorizontalize.neighbor(pitches,scale = scale)
            if i in [2]:
                new_pitches = ChordHorizontalize.neighbor_2(pitches,scale=scale)
            for pitch in new_pitches:
                note = m21.note.Note(pitch)
                note.duration.quarterLength = 0.25
                new_measure.append(note)
        return new_measure

    @staticmethod
    def pattern4(chords):
        new_measure = m21.stream.Measure()
        all_pitches = [chord.pitches for chord in chords]
        for i, pitches in enumerate(all_pitches):
            if i in [0]:
                new_pitches = ChordHorizontalize.arp(pitches, [2,0])
                ql = 0.5
            if i in [1]:
                new_pitches = ChordHorizontalize.arp(pitches, [0,1,2,1])
                ql = 0.25
            if i in [2]:
                new_pitches = ChordHorizontalize.arp(pitches, [0,1,2,1])
                ql= 0.25
            for pitch in new_pitches:
                note = m21.note.Note(pitch)
                note.duration.quarterLength = ql
                new_measure.append(note)
        return new_measure

    @staticmethod
    def pattern5(chords):
        new_measure = m21.stream.Measure()
        all_pitches = [chord.pitches for chord in chords]
        for i, pitches in enumerate(all_pitches):
            if i in [0]:
                new_pitches = ChordHorizontalize.neighbor(pitches, scale=scale)
                ql= 0.25
            if i in [1]:
                new_pitches = ChordHorizontalize.arp(pitches, [1, 2])
                ql = 0.5
            if i in [2]:
                new_pitches = ChordHorizontalize.arp(pitches, [1, 2])
                ql = 0.5
            for pitch in new_pitches:
                note = m21.note.Note(pitch)
                note.duration.quarterLength = ql
                new_measure.append(note)
        return new_measure




arp = ChordHorizontalize.arp
l_prog = ChordHorizontalize.linear_prog
hori_sequence = [l_prog, l_prog, l_prog]


def main():
    polyphony_path = '../data/Generated_xml/I I V I (no overlapping).mxl'

    stream = m21.converter.parse(polyphony_path)

    chords = stream.chordify()

    # chords = list(chords.getElementsByClass(m21.chord.Chord))
    melody = m21.stream.Stream()
    melody.insert(m21.tempo.MetronomeMark(80))
    melody.insert(m21.meter.TimeSignature('3/4'))
    measures = list(chords.getElementsByClass(m21.stream.Measure))
    for i, measure in enumerate(measures):
        if i < len(measures)-1:
            _chords = measure.getElementsByClass(m21.stream.chord.Chord)
            pattern = random.choice([BarHorizontalize.pattern5,BarHorizontalize.pattern5])
            new_measure = pattern(_chords)
            melody.append(new_measure)
        else:
            _chords = measure.getElementsByClass(m21.stream.chord.Chord)
            pattern = random.choice([BarHorizontalize.pattern5, BarHorizontalize.pattern5])
            new_measure = pattern(_chords)
            last_note = _chords[-1][2]
            last_note.duration.quarterLength=2
            new_measure.append(last_note)
            melody.append(new_measure)
    melody.show('text')
    melody.show()


if __name__ == '__main__':
    main()
