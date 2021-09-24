import collections
import copy
import itertools
import random
import sys
import time

import ipywidgets
import matplotlib.pyplot as plt
import music21.stream
import sklearn.cluster as cluster
import sklearn.manifold
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import music21 as m21
import os
import scipy.spatial.distance as distance
import plotly.graph_objects as go
import plotly.subplots
import scipy.fftpack
from ipywidgets import HTML, HBox, VBox

Mode = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 9, 10, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
}


class BarPatternFeatures:
    @staticmethod
    def scale_degree_and_duration(measure: music21.stream.Measure, key: music21.key.KeySignature, n_beat) -> np.ndarray:
        # array = np.zeros(shape=(n_beat,4,1,1))
        notes = list(measure.getElementsByClass(m21.note.Note))
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.duration.quarterLength] for note in notes]
        return notes

    @staticmethod
    def scale_degree_and_onset(measure: music21.stream.Measure, key: music21.key.KeySignature, n_beat) -> np.ndarray:
        # array = np.zeros(shape=(n_beat,4,1,1))
        notes = list(measure.getElementsByClass(m21.note.Note))
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.offset] for note in notes]
        return notes

    @staticmethod
    def scale_degree_and_onset_and_duration(measure: music21.stream.Measure, key: music21.key.KeySignature,
                                            n_beat) -> np.ndarray:
        # array = np.zeros(shape=(n_beat,4,1,1))
        notes = list(measure.getElementsByClass(m21.note.Note))
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.offset, note.duration.quarterLength] for note in
                 notes]
        return notes

    @staticmethod
    def scale_degree_and_onset_and_duration_and_contour(measure: music21.stream.Measure, key: music21.key.KeySignature,
                                                        n_beat) -> np.ndarray:
        # array = np.zeros(shape=(n_beat,4,1,1))
        notes = list(measure.getElementsByClass(m21.note.Note))
        scale_degrees = [note.pitch.pitchClass - key.tonic.pitchClass for note in notes]
        diff_scale_degrees = [0]
        diff_scale_degrees.extend([scale_degrees[i] - scale_degrees[i - 1] for i in range(1, len(scale_degrees))])
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.offset, note.duration.quarterLength,
                  diff_scale_degrees[i]] for i, note in
                 enumerate(notes)]
        return notes

    @staticmethod
    def rhythm_sixteenth_grid(measure: music21.stream.Measure, key: music21.key.KeySignature,
                              n_beat) -> np.ndarray:
        # array = np.zeros(shape=(n_beat,4,1,1))
        potential_voices = list(measure.getElementsByClass(m21.stream.Voice))
        if potential_voices != []:
            next_level = potential_voices[0]
        else:
            next_level = measure
        events = list(next_level.getElementsByClass([m21.note.Note, m21.chord.Chord]))
        # print('events: ',events)
        notes = []
        for event in events:
            if type(event) == m21.chord.Chord:
                note = event.notes[0]
                note.offset = event.offset
                # print(note.offset)
                notes.append(note)
            elif type(event) == m21.note.Note:
                note = event
                notes.append(note)
            else:
                # print('encountered neither chord or note: ', type(event), 'disgard event')
                pass
        # print('notes:', notes)

        notes = [[note.pitch.midi - key.tonic.midi, note.offset, note.duration.quarterLength] for i, note in
                 enumerate(notes)]
        grid = np.full(12, fill_value='_', dtype=str)
        for i, note in enumerate(notes):
            index = int(note[1] * 4)
            if index >= 12:
                # print('encountered a non 3/4 bar')
                pass
            else:
                grid[index] = 'x'
                duration = int(note[2] * 4)
                grid[index + 1:index + duration + 1] = '-'
        return grid

    @staticmethod
    def rhythm_incremental(measure: music21.stream.Measure, key: music21.key.KeySignature,
                           n_beat) -> np.ndarray:
        rhythm_grid = BarPatternFeatures.rhythm_sixteenth_grid(measure, key,
                                                               n_beat)
        indices_4 = 4 * np.concatenate(np.tile([0, 1, 2], (4, 1)).T)
        rhythm_4 = rhythm_grid[indices_4]
        rhythm_4[np.concatenate([np.array([1, 2, 3]) + 4 * i for i in range(3)])] = '-'

        indices_8 = 2 * np.concatenate(np.tile([0, 1, 2, 3, 4, 5], (2, 1)).T)
        rhythm_8 = rhythm_grid[indices_8]
        rhythm_8[np.concatenate([np.array([1, 3]) + 4 * i for i in range(3)])] = '-'

        rhythm_16 = rhythm_grid

        rhythms = [rhythm_4, rhythm_8, rhythm_16]
        return rhythms

    @staticmethod
    def contour_sixteenth_grid(measure: music21.stream.Measure, key: music21.key.KeySignature,
                               n_beat, n_grid=12) -> np.ndarray:
        # array = np.zeros(shape=(n_beat,4,1,1))
        potential_voices = list(measure.getElementsByClass(m21.stream.Voice))
        if potential_voices != []:
            next_level = potential_voices[0]
        else:
            next_level = measure
        events = list(next_level.getElementsByClass([m21.note.Note, m21.chord.Chord]))
        # print('events: ',events)
        notes = []
        for event in events:
            if type(event) == m21.chord.Chord:
                note = event.notes[-1]
                note.offset = event.offset
                # print(note.offset)
                notes.append(note)
            elif type(event) == m21.note.Note:
                note = event
                notes.append(note)
            else:
                # print('encountered neither chord or note: ', type(event), 'disgard event')
                pass
        # print('notes:',notes)

        notes = [[note.pitch.midi - key.tonic.midi, note.offset, note.duration.quarterLength] for i, note in
                 enumerate(notes)]

        # print('notes:', notes)
        grid = np.full(n_grid, fill_value=999, dtype=object)
        for i, note in enumerate(notes):
            index = int(note[1] * 4)
            if index >= 12:
                # print('encountered a non 3/4 bar, disgard')
                pass
            else:
                refined_index = index * int(n_grid / 12)
                # print('refined_index: ', refined_index)
                grid[refined_index] = note[0]
                duration = int(note[2] * 4 * n_grid / 12)
                # print('duration: ',duration)
                refined_duration = int(duration * n_grid / 12)
                grid[refined_index + 1:refined_index + refined_duration + 1] = note[0]
        for i, x in enumerate(grid):
            if x == 999:
                grid[i] = grid[i - 1]
        if np.count_nonzero(grid == 999) > 0:
            if np.alltrue(grid == 999):
                pass
            else:
                # print('encountered rest in grid: ', grid)
                pass
        return grid

    @staticmethod
    def contour_cosine(measure: music21.stream.Measure, key: music21.key.KeySignature,
                       n_beat) -> np.ndarray:
        ## cosine
        contour_grid = BarPatternFeatures.contour_sixteenth_grid(measure, key, n_beat, n_grid=120)
        dct = scipy.fftpack.dct(contour_grid, norm='forward')
        dct = dct[0:25]
        return dct

    @staticmethod
    def contour_cosine_downbeat(measure: music21.stream.Measure, key: music21.key.KeySignature,
                                n_beat) -> np.ndarray:
        ## cosine
        contour_grid = BarPatternFeatures.contour_sixteenth_grid(measure, key, n_beat, n_grid=120)
        indices = 40 * np.concatenate(np.tile([0, 1, 2], (40, 1)).T)
        contour_grid_downbeat = contour_grid[indices]

        dct = scipy.fftpack.dct(contour_grid_downbeat, norm='forward')
        dct = dct[:25]
        return dct

    @staticmethod
    def contour_cosine_eighth(measure: music21.stream.Measure, key: music21.key.KeySignature,
                              n_beat) -> np.ndarray:
        ## cosine
        contour_grid = BarPatternFeatures.contour_sixteenth_grid(measure, key, n_beat, n_grid=120)
        indices = 20 * np.concatenate(np.tile([0, 1, 2, 3, 4, 5], (20, 1)).T)
        contour_grid_downbeat = contour_grid[indices]

        dct = scipy.fftpack.dct(contour_grid_downbeat, norm='forward')
        dct = dct[:25]
        return dct

    @staticmethod
    def texture_contour_diff(measure: (m21.stream.Measure, m21.key.KeySignature, m21.meter.TimeSignature)):
        # print('measure: ',measure)
        contour_detailed_16th = BarPatternFeatures.contour_cosine(measure[0], key=measure[1],
                                                                  n_beat=measure[2].numerator)

        contour_downbeat = BarPatternFeatures.contour_cosine_downbeat(measure[0], key=measure[1],
                                                                      n_beat=measure[2].numerator)
        contour_detailed_8th = BarPatternFeatures.contour_cosine_eighth(measure[0], key=measure[1],
                                                                        n_beat=measure[2].numerator)

        pitches = BarPatternFeatures.contour_sixteenth_grid(measure[0], key=measure[1],
                                                            n_beat=measure[2].numerator, n_grid=300)
        contour_coe_diff_4_8 = contour_detailed_8th - contour_downbeat
        contour_coe_diff_16_8 = contour_detailed_16th - contour_detailed_8th

        contour_approxs_16th = scipy.fftpack.idct(contour_detailed_16th, norm='backward', n=300)
        contour_approxs_downbeat = scipy.fftpack.idct(contour_downbeat, norm='backward', n=300)
        contour_approxs_8th = scipy.fftpack.idct(contour_detailed_8th, norm='backward', n=300)
        contour_diff_4_8 = contour_approxs_8th - contour_approxs_downbeat
        contour_diff_16_8 = contour_approxs_16th - contour_approxs_8th

        contour = [contour_approxs_downbeat, contour_diff_4_8, contour_diff_16_8]
        coeff = [contour_downbeat, contour_coe_diff_4_8, contour_coe_diff_16_8]

        return pitches, contour, coeff

    @staticmethod
    def batch_texture_contour_diff(all_measures: list[m21.stream.Measure]):
        all_pitches = []
        all_contour = []
        all_coeff = []
        for measure in all_measures:
            pitches, contour, coeff = BarPatternFeatures.texture_contour_diff(measure)

            coeff = np.array(coeff)
            contour = np.array(contour)
            # print('contour: ', contour)
            contour = np.array([np.sum(contour[0:i + 1], axis=0) for i, x in enumerate(contour)])
            # print('accu_contour: ', contour)
            all_contour.append(contour)
            all_coeff.append(coeff)
            all_pitches.append(pitches)
        all_contour = np.array(all_contour)
        all_coeff = np.array(all_coeff)
        all_pitches = np.array(all_pitches)
        return all_pitches, all_contour, all_coeff


class PieceFeatures:
    def __init__(self,cadential_features):
        self.cadential_features = cadential_features
    @staticmethod
    def clean_measures(piece: list[(object, object, object, object)]):
        cleaned_piece = []
        for x in piece:
            m21_measure = x[0]
            duration = m21_measure.duration.quarterLength
            if duration >= 2.0:
                cleaned_piece.append(x)
            else:
                print('ignore measure with duration ', duration)
        return cleaned_piece

    def get_location_contour_rhythm_dict(self, piece: list[(object, object, object, object)], feature_evaluator):
        piece = PieceFeatures.clean_measures(piece)
        list_of_feature_dict = []
        cadence_locations = PieceFeatures.cadence_detector(piece, feature_evaluator , self.cadential_features)
        partition_by_cadence = np.diff(np.concatenate([[0], cadence_locations + 1]))
        print(
            '{:>12} {:>12} | {:>12} {:>12} | {:>12} {:>12}'.format('partition_by_cadence: ', str(partition_by_cadence),
                                                                   'cadence_locations: ', str(cadence_locations),
                                                                   'piece_length:', str(len(piece))))
        if len(cadence_locations) == 0:
            StreamBuilder.measures_to_stream(piece, title=piece[-1][-1])
        location_numerator = 0
        cadence_passed = 0
        for i, (measure, key_signature, time_signature, file_name) in enumerate(piece):
            if i in cadence_locations + 1:
                cadence_passed = cadence_passed + 1
                location_numerator = 0
            else:
                if i == 0:
                    location_numerator = 0
                else:
                    location_numerator = location_numerator + 1

            location_denominator = partition_by_cadence[cadence_passed]
            # print('location_numerator: ', location_numerator, 'location_denominator: ', location_denominator)
            location = [location_numerator, location_denominator]
            contour = BarPatternFeatures.contour_cosine(measure, key_signature, n_beat=time_signature.numerator)
            rhythm = BarPatternFeatures.rhythm_sixteenth_grid(measure, key_signature, n_beat=time_signature.numerator)
            feature = {'location': location, 'contour': contour, 'rhythm': rhythm, 'key': key_signature,
                       'source': file_name.replace('../data/xml/', '')}
            list_of_feature_dict.append(feature)
        return list_of_feature_dict

    @staticmethod
    def get_contour_rhythm_dict(piece: list[(object, object, object, object)]):
        list_of_feature_dict = []

        for i, (measure, key_signature, time_signature, file_name) in enumerate(piece):
            contour = BarPatternFeatures.contour_cosine(measure, key_signature, n_beat=time_signature.numerator)
            rhythm = BarPatternFeatures.rhythm_sixteenth_grid(measure, key_signature, n_beat=time_signature.numerator)
            feature = {'contour': contour, 'rhythm': rhythm, 'key': key_signature, 'source': file_name}
            list_of_feature_dict.append(feature)
        return list_of_feature_dict

    @staticmethod
    def cadence_detector(piece: list[(object, object, object, object)], feature_evaluator, cadential_features):
        cadence_matchness = []
        for i, (measure, key_signature, time_signature, file_name) in enumerate(piece):
            contour = BarPatternFeatures.contour_cosine(measure, key_signature, n_beat=time_signature.numerator)
            rhythm = BarPatternFeatures.rhythm_sixteenth_grid(measure, key_signature, n_beat=time_signature.numerator)
            feature = {'contour': contour, 'rhythm': rhythm}
            #cadence_features = PieceFeatures.get_contour_rhythm_dict(experiment.strong_candeces)
            cadences_match = np.max(
                [feature_evaluator._match_with_latent(feature, cadence_feature) for cadence_feature in
                 cadential_features])
            # print('cadences_match: ',cadences_match)
            cadence_matchness.append(cadences_match)
        cadence_matchness = np.array(cadence_matchness)
        # exp_cadence_matchness = np.exp(cadence_matchness)
        # softmax_cadence_matchness = exp_cadence_matchness / np.sum(exp_cadence_matchness)
        # print(softmax_cadence_matchness)
        # print(cadence_matchness)
        candidate_cadence_location = np.where(cadence_matchness > 0.5)[0]

        # xs = np.arange(len(cadence_matchness))+1
        # plt.plot(xs,cadence_matchness,marker='o',zorder=0)
        # plt.title('cadence-like of each measure')
        # plt.xticks(xs)
        # print(# cadence_matchness)
        # print(np.argsort(-cadence_matchness))
        # plt.scatter(candidate_cadence_location+1,cadence_matchness[candidate_cadence_location],marker='x',c='red',zorder=1)
        # StreamBuilder.measures_to_stream(piece)
        # print(np.where(cadence_matchness>0.5))
        # plt.show()
        return candidate_cadence_location


class DistanceFunction:
    @staticmethod
    def levenshtein(_seq1, _seq2):
        seq1 = _seq1[0]
        seq2 = _seq2[0]
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y), dtype=object)
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = matrix[x - 1, y - 1]
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        return matrix[size_x - 1, size_y - 1]

    @staticmethod
    def levenshtein_with_operation_cost(_seq1, _seq2):
        def insert_cost(seq):
            duration = seq[2]
            return (4 * duration) * (4 * duration)

        def substitution_cost(seq1, seq2):
            pitch_diff = max(1, abs(seq2[0] - seq1[0]))
            duration_diff = max(1, 4 * abs(seq2[2] - seq1[2]))
            return pitch_diff * duration_diff

        def deletion_cost(seq):
            duration = seq[2]
            return (4 * duration) * (4 * duration)

        costs_dict = {
            'insert': insert_cost,
            'substitute': substitution_cost,
            'delete': deletion_cost
        }
        seq1 = _seq1[0]
        seq2 = _seq2[0]
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y), dtype=object)
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = matrix[x - 1, y - 1]
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + costs_dict['insert'](_seq2[0][y - 1]),
                        matrix[x - 1, y - 1] + costs_dict['substitute'](_seq1[0][x - 1], _seq2[0][y - 1]),
                        matrix[x, y - 1] + costs_dict['delete'](_seq1[0][x - 1]),
                    )
        return matrix[size_x - 1, size_y - 1]

    @staticmethod
    def levenshtein_with_operation_cost_pitch_onset_duration(_seq1, _seq2):
        def insert_cost(seq):
            duration = seq[2]
            return (4 * duration) ** 2

        def substitution_cost(seq1, seq2):
            pitch_diff = 10 * abs(seq2[0] - seq1[0])
            onset_diff = 4 * abs(seq2[1] - seq1[1])
            return 1.3 * pitch_diff ** 2 * onset_diff

        def deletion_cost(seq):
            duration = seq[2]
            return (4 * duration) ** 2

        costs_dict = {
            'insert': insert_cost,
            'substitute': substitution_cost,
            'delete': deletion_cost
        }
        seq1 = _seq1[0]
        seq2 = _seq2[0]
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y), dtype=object)
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = matrix[x - 1, y - 1]
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + costs_dict['insert'](_seq2[0][y - 1]),
                        matrix[x - 1, y - 1] + costs_dict['substitute'](_seq1[0][x - 1], _seq2[0][y - 1]),
                        matrix[x, y - 1] + costs_dict['delete'](_seq1[0][x - 1]),
                    )
        return matrix[size_x - 1, size_y - 1]

    @staticmethod
    def contour_distance(_seq1, _seq2):
        def insert_cost(seq):
            duration = seq[2]
            return 1

        def substitution_cost(seq1, seq2):
            pitch_diff = np.tanh(abs(seq2[0] - seq1[0]))
            onset_diff = 4 * abs(seq2[1] - seq1[1])
            contour_diff = np.tanh(abs(seq2[3] - seq1[3]))
            return contour_diff

        def deletion_cost(seq):
            duration = seq[2]
            return 1

        costs_dict = {
            'insert': insert_cost,
            'substitute': substitution_cost,
            'delete': deletion_cost
        }
        seq1 = _seq1[0]
        seq2 = _seq2[0]
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y), dtype=object)
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = matrix[x - 1, y - 1]
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + costs_dict['insert'](_seq2[0][y - 1]),
                        matrix[x - 1, y - 1] + costs_dict['substitute'](_seq1[0][x - 1], _seq2[0][y - 1]),
                        matrix[x, y - 1] + costs_dict['delete'](_seq1[0][x - 1]),
                    )
        return matrix[size_x - 1, size_y - 1]

    @staticmethod
    def rhythm_grid_distance(_seq1, _seq2):
        def insert_cost(char):
            return 1

        def substitution_cost(char1, char2):
            if set([char1, char2]) == set(['x', '_']):
                cost = 1.3
            else:
                cost = 1.3
            return cost

        def deletion_cost(char):
            return 1

        costs_dict = {
            'insert': insert_cost,
            'substitute': substitution_cost,
            'delete': deletion_cost
        }

        size_x = len(_seq1) + 1
        size_y = len(_seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if _seq1[x - 1] == _seq2[y - 1]:
                    matrix[x, y] = matrix[x - 1, y - 1]
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + costs_dict['insert'](_seq2[y - 1]),
                        matrix[x - 1, y - 1] + costs_dict['substitute'](_seq1[x - 1], _seq2[y - 1]),
                        matrix[x, y - 1] + costs_dict['delete'](_seq1[x - 1]),
                    )
        return matrix[size_x - 1, size_y - 1]

    @staticmethod
    def contour_grid_distance(_seq1, _seq2):
        def insert_cost(char):
            return 1

        def substitution_cost(char1, char2):
            cost = abs(char2 - char1)
            return cost

        def deletion_cost(char):
            return 1

        costs_dict = {
            'insert': insert_cost,
            'substitute': substitution_cost,
            'delete': deletion_cost
        }

        size_x = len(_seq1) + 1
        size_y = len(_seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if _seq1[x - 1] == _seq2[y - 1]:
                    matrix[x, y] = matrix[x - 1, y - 1]
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + costs_dict['insert'](_seq2[y - 1]),
                        matrix[x - 1, y - 1] + costs_dict['substitute'](_seq1[x - 1], _seq2[y - 1]),
                        matrix[x, y - 1] + costs_dict['delete'](_seq1[x - 1]),
                    )
        return matrix[size_x - 1, size_y - 1]


class Combination:
    @staticmethod
    def rhythm_intersection(rhythm1, rhythm2):
        """take the common onset"""
        assert len(rhythm1) == len(rhythm2)
        new_rhythm = []
        for i, (r1, r2) in enumerate(zip(rhythm1, rhythm2)):
            set_of_char = {r1, r2}
            if r1 == r2:
                new_rhythm.append(r1)
            elif '_' in set_of_char:
                new_rhythm.append('_')
            elif '-' in set_of_char:
                new_rhythm.append('-')
            else:
                print('something wrong during rhythm_intersection', rhythm1, rhythm2)
        return new_rhythm

    @staticmethod
    def rhythm_union(rhythm1, rhythm2):
        """take the common onset"""
        assert len(rhythm1) == len(rhythm2)
        new_rhythm = []
        for i, (r1, r2) in enumerate(zip(rhythm1, rhythm2)):
            set_of_char = {r1, r2}
            if r1 == r2:
                new_rhythm.append(r1)
            elif 'x' in set_of_char:
                new_rhythm.append('x')
            elif '-' in set_of_char:
                new_rhythm.append('-')
            else:
                print('something wrong during rhythm_intersection', rhythm1, rhythm2)
        print('rhythm1: ', rhythm1)
        print('rhythm2: ', rhythm2)
        print(new_rhythm)
        return new_rhythm


class Plot:
    @staticmethod
    def draw_scatter(distance_matrix, text_arrays, file_name, model):
        if type(distance_matrix) == list:
            subplot_distance_matrices = distance_matrix
        else:
            subplot_distance_matrices = [distance_matrix]
        if type(model) == list:
            subplot_models = model
        else:
            subplot_models = [model]
        plotly.io.templates.default = "plotly_white"

        fig = plotly.subplots.make_subplots(rows=1, cols=len(subplot_distance_matrices),
                                            specs=[[{"type": "scene"}, {"type": "scene"}],
                                                   ])
        for i, (distance_matrix, model) in enumerate(zip(subplot_distance_matrices, subplot_models)):
            tsne = sklearn.manifold.TSNE(n_components=3, metric="precomputed", perplexity=40, learning_rate=20,
                                         square_distances=True)
            embeded = tsne.fit_transform(distance_matrix)
            # text_list = np.squeeze(text_arrays)
            # text_list = ['<br>'.join([str(y) for y in x]) for x in text_arrays]
            # text_list = [str(np.array(x)).replace('\n', '<br>') for x in text_list]
            # print(text_arrays[:10])
            # text_list = ['<br>'.join(x) for x in text_arrays]
            text_list = text_arrays

            scatter = go.Scatter3d(x=embeded[:, 0], y=embeded[:, 1], z=embeded[:, 2], mode='markers',
                                   marker=dict(
                                       size=8,
                                       color=model.labels_,  # set color to an array/list of desired values
                                       colorscale='Phase',
                                       symbol='circle',
                                       opacity=0.4
                                   ), hovertemplate='%{text}',
                                   text=text_list, textfont=dict({'family': 'Roboto Mono'}))

            # fig = go.FigureWidget()

            def update_point(trace, points, selector):
                c = list(trace.marker.color)
                s = list(trace.marker.size)
                print('updating')
                for i in points.point_inds:
                    c[i] = '#bae2be'
                    s[i] = 20
                    with fig.batch_update():
                        trace.marker.color = c
                        trace.marker.size = s

            scatter.on_click(update_point)
            fig.add_trace(scatter, row=1, col=i + 1)

        fig.write_html(file_name + '.html')

    @staticmethod
    def draw_scatter_no_TSNE(triples, text_arrays, file_name, model):
        plotly.io.templates.default = "plotly_white"

        fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                            specs=[[{"type": "scene"}],
                                                   ])

        # text_list = np.squeeze(text_arrays)
        text_list = [' '.join([str(y) for y in x]) for x in text_arrays]
        # text_list = [str(np.array(x)).replace('\n', '<br>') for x in text_list]
        print('triples.shape: ', triples.shape)
        scatter = go.Scatter3d(x=triples[:, 0], y=triples[:, 1], z=triples[:, 2], mode='markers',
                               marker=dict(
                                   size=6,
                                   color=model.labels_,  # set color to an array/list of desired values
                                   colorscale='Phase',
                                   opacity=1
                               ), hovertemplate='%{text}',
                               text=text_list)

        def update_point(trace, points, selector):
            c = list(scatter.marker.color)
            s = list(scatter.marker.size)
            print('updating')
            for i in points.point_inds:
                c[i] = '#bae2be'
                s[i] = 20
                with fig.batch_update():
                    scatter.marker.color = c
                    scatter.marker.size = s

        scatter.on_click(update_point)
        fig = go.FigureWidget()
        fig.add_trace(scatter)
        fig.write_html(file_name + '.html')

    @staticmethod
    def draw_heatmap(distance_matrix, measure_feature):
        plotly.io.templates.default = "plotly_white"

        fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                            specs=[[{"type": "scene"}],
                                                   ])
        display_array = np.squeeze(measure_feature).tolist()
        hovertext = list()
        for yi, yy in enumerate(display_array):
            hovertext.append(list())
            for xi, xx in enumerate(display_array):
                hovertext[-1].append(
                    'bar 1: {}<br />bar 2: {}<br />distance: {}'.format(xx, yy, distance_matrix[yi][xi]))
        fig.add_trace(
            go.Heatmap(z=distance_matrix, hovertemplate='%{text}<extra></extra>', text=hovertext, showscale=False),
            row=1, col=1
        )
        fig.show()

    @staticmethod
    def draw_contour(pitch, contour_approxs, n_sample_points):
        n = len(pitch)
        print(n)
        fig, axs = plt.subplots(10, 10)

        for i, ax in enumerate(axs.reshape(-1)):
            if i < n:

                xs = np.linspace(0, 11, num=n_sample_points)

                ax.scatter(xs, pitch[i], marker='s', c='grey', s=5)

                colors = ['red', 'green', 'blue', 'orange', 'grey', 'purple']
                for j, contour_approx in enumerate(contour_approxs[i]):
                    ax.plot(xs, contour_approx, c=colors[j % len(colors)], alpha=0.5)

        plt.show()

    @staticmethod
    def draw_contour_minimal(contour_approx):

        fig, ax = plt.subplots()
        n_sample_points = contour_approx.shape[0]
        xs = np.linspace(0, 12, num=n_sample_points, endpoint=False)
        ax.set_xticks(np.arange(12))
        ax.grid()
        ax.plot(xs, contour_approx, c='red', alpha=0.5)
        plt.show()

    @staticmethod
    def draw_dct(dcts):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 2)
        ax.plot(np.average(np.abs(dcts), axis=0))
        plt.show()


class FeatureEvaluation:
    contour_distance_func = lambda x, y, axis=None: np.linalg.norm(x - y, axis=axis, ord=2)
    rhythm_distance_func = DistanceFunction.rhythm_grid_distance

    def __init__(self, joint_distribution_of_features: list[dict],
                 contour_distance_func: callable = contour_distance_func,
                 rhythm_distance_func: callable = rhythm_distance_func, ):
        self.contour_distance_func = contour_distance_func
        self.rhythm_distance_func = rhythm_distance_func
        self.joint_distribution_of_features = joint_distribution_of_features
        self.locations_of_all_rhythms = None
        self.locations_of_all_contours = None
        self.rhythms_of_all_contours = None
        self.contours_of_all_rhythms = None

    def get_locations_of_all_rhythms(self):

        locations_of_all_rhythms = dict()

        for d in self.joint_distribution_of_features:
            rhythm = d['rhythm']
            location = d['location']
            if str(rhythm) not in locations_of_all_rhythms.keys():
                locations_of_all_rhythms.update({str(rhythm): [location]})
            else:
                locations_of_all_rhythms[str(rhythm)].append(location)

        return locations_of_all_rhythms

    def get_locations_of_all_contours(self):
        locations_of_all_contours = dict()
        for d in self.joint_distribution_of_features:
            contour = d['contour']
            location = d['location']
            if str(contour) not in locations_of_all_contours.keys():
                locations_of_all_contours.update({str(contour): [location]})
            else:
                locations_of_all_contours[str(contour)].append(location)

        return locations_of_all_contours

    def get_rhythms_of_all_contours(self):
        rhythms_of_all_contours = dict()
        for d in self.joint_distribution_of_features:
            contour = d['contour']
            rhythm = d['rhythm']
            if str(contour) not in rhythms_of_all_contours.keys():
                rhythms_of_all_contours.update({str(contour): [rhythm]})
            else:
                rhythms_of_all_contours[str(contour)].append(rhythm)

        return rhythms_of_all_contours

    def get_contours_of_all_rhythms(self):
        contours_of_all_rhythms = dict()
        for d in self.joint_distribution_of_features:
            rhythm = d['rhythm']
            contour = d['contour']
            if str(rhythm) not in contours_of_all_rhythms.keys():
                contours_of_all_rhythms.update({str(rhythm): [contour]})
            else:
                contours_of_all_rhythms[str(rhythm)].append(contour)

        return contours_of_all_rhythms

    def location_match(self, features, current_location):
        if self.locations_of_all_rhythms == None:
            self.locations_of_all_rhythms = self.get_locations_of_all_rhythms()
        if self.locations_of_all_contours == None:
            self.locations_of_all_contours = self.get_locations_of_all_contours()

        rhythm = features['rhythm']
        contour = features['contour']
        locations_of_this_rhythm = np.array(self.locations_of_all_rhythms[str(rhythm)])
        locations_of_this_contour = np.array(self.locations_of_all_contours[str(contour)])
        # locations_of_this_rhythm = np.array([d['location'] for d in self.joint_distribution_of_features if np.array_equal(d['rhythm'],rhythm)])
        # locations_of_this_contour = np.array([d['location'] for d in self.joint_distribution_of_features if np.array_equal(d['contour'],contour)])
        # print('locations_of_this_rhythm: ',locations_of_this_rhythm)
        # print('locations_of_this_contour: ',locations_of_this_contour)
        # print('current_location: ',current_location)
        rhythm_location_distances = np.sum(np.abs(locations_of_this_rhythm - current_location), axis=-1)
        contour_location_distances = np.sum(np.abs(locations_of_this_contour - current_location), axis=-1)
        # print('np.array(locations_of_this_rhythm): ',np.array(locations_of_this_rhythm))
        # print('rhythm_location_distances: ',np.abs(np.array(locations_of_this_rhythm) - np.array(current_location)))
        min_rhythm_location_distance = np.min(rhythm_location_distances)
        min_contour_location_distance = np.min(contour_location_distances)
        total_distance = np.linalg.norm(np.array([min_contour_location_distance, min_rhythm_location_distance]), ord=2)
        plausibility = 1 / (1 + total_distance)
        return plausibility

    def cadence_only_location_match(self, features, current_location):
        if self.locations_of_all_rhythms == None:
            self.locations_of_all_rhythms = self.get_locations_of_all_rhythms()
        if self.locations_of_all_contours == None:
            self.locations_of_all_contours = self.get_locations_of_all_contours()

        rhythm = features['rhythm']
        contour = features['contour']
        locations_of_this_rhythm = np.array(self.locations_of_all_rhythms[str(rhythm)])
        locations_of_this_contour = np.array(self.locations_of_all_contours[str(contour)])

        if current_location[0] != current_location[-1] - 1:
            rhythm_location_distances = np.average(locations_of_this_rhythm[:, -1] - locations_of_this_rhythm[:, 0] > 1)
            contour_location_distances = np.average(
                locations_of_this_contour[:, -1] - locations_of_this_contour[:, 0] > 1)
            plausibility = (rhythm_location_distances + contour_location_distances) / 2
        else:
            rhythm_location_distances = np.sum(np.abs(locations_of_this_rhythm - current_location), axis=-1)
            contour_location_distances = np.sum(np.abs(locations_of_this_contour - current_location), axis=-1)
            min_rhythm_location_distance = np.min(rhythm_location_distances)
            min_contour_location_distance = np.min(contour_location_distances)
            total_distance = np.linalg.norm(np.array([min_contour_location_distance, min_rhythm_location_distance]),
                                            ord=2)
            plausibility = 1 / (1 + total_distance)
        return plausibility

    def internal_match(self, features):
        if self.rhythms_of_all_contours == None:
            self.rhythms_of_all_contours = self.get_rhythms_of_all_contours()
        if self.contours_of_all_rhythms == None:
            self.contours_of_all_rhythms = self.get_contours_of_all_rhythms()

        rhythm = features['rhythm']
        contour = features['contour']

        # rhythms_contour = np.array(
        #    [d['rhythm'] for d in self.joint_distribution_of_features if np.array_equal(d['contour'], contour)])
        # contours_rhythm = np.array(
        #    [d['contour'] for d in self.joint_distribution_of_features if np.array_equal(d['rhythm'], rhythm)])

        rhythms_contour = np.array(self.rhythms_of_all_contours[str(contour)])
        contours_rhythm = np.array(self.contours_of_all_rhythms[str(rhythm)])

        contour_distances = self.contour_distance_func(contours_rhythm, contour, axis=-1)
        assert contours_rhythm.shape[0] == contour_distances.shape[0], (contours_rhythm.shape, contour_distances.shape)
        min_distance_contour = np.min(contour_distances, axis=0)
        min_distance_rhythm = np.min(
            np.abs([self.rhythm_distance_func(rhythm_contour, rhythm) for rhythm_contour in rhythms_contour]))

        # min_distance_contour = np.min(np.abs([self.contour_distance_func(contour_rhythm, contour) for contour_rhythm in contours_rhythm]))
        # min_distance_rhythm = np.min(np.abs([self.rhythm_distance_func(rhythm_contour, rhythm) for rhythm_contour in rhythms_contour]))

        total_distance = np.linalg.norm(np.array([min_distance_contour, min_distance_rhythm]), ord=2)
        # total_distance = np.sum([min_distance_contour, min_distance_rhythm])/2
        plausibility = 1 / (1 + total_distance)
        return plausibility

    def match_with_latent(self, features, latent_features):
        if latent_features == None:
            # print('latent_features is None')
            plausibility = 1
        else:

            rhythm = features['rhythm']
            contour = features['contour']
            nan_contour = np.zeros(25)
            nan_contour[:] = np.NAN
            empty_contour = np.ma.masked_array(nan_contour, mask=np.ones(25))
            # print('np.ma.getmask(empty_contour): ',np.ma.getmask(empty_contour))
            latent_rhythms = [latent_feature['rhythm'] if type(latent_feature) == dict else None for latent_feature in
                              latent_features]
            latent_contours = np.ma.masked_values(
                [latent_feature['contour'] if type(latent_feature) == dict else empty_contour for latent_feature in
                 latent_features], np.NaN)
            # print('latent_contours.shape: ',latent_contours.shape)
            # print('np.ma.getmask(latent_contours): ',np.ma.getmask(latent_contours))

            distance_contour = self.contour_distance_func(contour, latent_contours, axis=-1)
            distance_contour[np.isnan(distance_contour)] = 0

            # print('distance_contour.shape: ',distance_contour.shape)
            # print('distance_contour: ', distance_contour)
            distance_rhythm = np.array(
                [self.rhythm_distance_func(rhythm, latent_rhythm) if type(latent_rhythm) is np.ndarray else 0 for
                 latent_rhythm in latent_rhythms])

            # print('distance_contour.shape: ',distance_contour.shape)
            # print('distance_rhythm.shape: ',distance_rhythm.shape)

            stacked_distances = np.stack([distance_contour, distance_rhythm], axis=1)
            # print('stacked_distances.shape: ',stacked_distances.shape)
            total_distance = np.linalg.norm(stacked_distances, ord=2, axis=1)
            # print('total_distance.shape',total_distance.shape)
            plausibility = 1 / (1 + total_distance)
        return plausibility

    def _match_with_latent(self, features, latent_features):
        if latent_features == None:
            # print('latent_features is None')
            plausibility = 1
        else:
            rhythm = features['rhythm']
            contour = features['contour']

            latent_rhythm = latent_features['rhythm']
            latent_contour = latent_features['contour']

            distance_rhythm = self.rhythm_distance_func(rhythm, latent_rhythm)
            distance_contour = self.contour_distance_func(contour, latent_contour)

            # total_distance = np.linalg.norm(np.array([distance_contour, distance_rhythm]), ord=2)
            weights = np.array([1, 1])
            distances = np.array([distance_contour, distance_rhythm])
            weights_distances = weights / np.sum(weights) * distances
            total_distance = np.sum(weights_distances)
            plausibility = 1 / (1 + total_distance)
        return plausibility


class MelodySynthesis:

    @staticmethod
    def combine_contour_rhythm_vl(contour_coeffs, rhythm_grid, vl, sample_point_size, pc_distribution_grid,
                                  type='coeff'):
        """use harmonic content ended in pc_distribution_grid, which has shape=(12,12), meaning (#16th-notes,#pc)"""
        sample_point_size = 1200
        if type == 'coeff':
            approx_contour = scipy.fftpack.idct(contour_coeffs, norm='backward', n=sample_point_size)
        if type == 'approx':
            approx_contour = contour_coeffs

        approx_contour_16_grid = []
        for i in range(12):
            xs = np.linspace(0, 12, num=sample_point_size, endpoint=False)
            lower_index = np.argmin(np.abs(xs - (i + 0.1)))
            upper_index = max(np.argmin(np.abs(xs - (i + 0.9))), lower_index + 1)
            approx_contour_pitch = np.average(approx_contour[lower_index:upper_index])
            approx_contour_16_grid.append(approx_contour_pitch)

        # print('approx_contour: ', approx_contour)

        pc_distribution_grid = pc_distribution_grid + 0
        scale_pc_distribution_grid = 1 * (pc_distribution_grid > 0)
        chord_pc_distribution_grid = 1 * (pc_distribution_grid > 1)
        off_beat_index = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11]).reshape(-1, 1)
        chord_pc_distribution_grid[off_beat_index] = 0
        eigth_off_beat_index = np.array([2,5,7,10]).reshape(-1, 1)
        chord_pc_distribution_grid[eigth_off_beat_index] = 0.5
        print('scale_pc_distribution_grid: ',scale_pc_distribution_grid)
        print('chord_pc_distribution_grid: ',chord_pc_distribution_grid)
        print('')
        # print('pc_distribution_grid: ',pc_distribution_grid)
        # pc_distribution_grid = np.array([1])+0.1
        # pc_distribution_grid = np.tile(pc_distribution_grid,(12,12))
        pitch_grid = []

        # print('approx_contour_16_grid: ', approx_contour_16_grid)
        # print('---')
        def pc_motion_to_scale_degree_motion(pc_motion, tiled_scale_pc_distribution_grid, current_pitch):
            current_pitch_index = np.argwhere(tiled_pitches == current_pitch).reshape(-1)[0]

            rounded_pc_motion = int(np.rint(pc_motion))
            # print('rounded_pc_motion: ',rounded_pc_motion)
            # print('current_pitch_index: ',current_pitch_index)
            if pc_motion > 0:
                scale_degree_motion = np.sum(
                    tiled_scale_pc_distribution_grid[current_pitch_index:current_pitch_index + rounded_pc_motion])
            if pc_motion == 0:
                scale_degree_motion = 0
            if pc_motion < 0:
                scale_degree_motion = -np.sum(
                    tiled_scale_pc_distribution_grid[current_pitch_index + rounded_pc_motion:current_pitch_index])
            return scale_degree_motion

        for i, x in enumerate(rhythm_grid):
            if x == 'x':
                tiled_pc_distribution_grid = np.tile(pc_distribution_grid[i], 4)  # shape = (3*12,)
                tiled_pitches = np.arange(tiled_pc_distribution_grid.shape[0]) - 12  # shape = (3*12,)
                diff = np.abs(tiled_pitches - approx_contour_16_grid[i])  # shape = (3*12,1)
                tiled_scale_pc_distribution_grid = np.tile(scale_pc_distribution_grid[i], 4)
                tiled_chord_pc_distribution_grid = np.tile(chord_pc_distribution_grid[i], 4)
                #print('tiled_scale_pc_distribution_grid: ',tiled_scale_pc_distribution_grid)
                #print('tiled_chord_pc_distribution_grid: ',tiled_chord_pc_distribution_grid)

                print('---------------')
                if i > 0:
                    target_derivative = approx_contour_16_grid[i] - approx_contour_16_grid[i - 1]
                    # target_derivative_in_degree = pc_motion_to_scale_degree_motion(target_derivative,
                    #                                                               tiled_scale_pc_distribution_grid,
                    #                                                               pitch_grid[-1])
                    # print('target_derivative_in_degree:',target_derivative_in_degree)
                    target_direction = 1 * (target_derivative > 0) + 0 * (np.abs(target_derivative) == 0) + -1 * (
                                target_derivative < 0)
                    # print('target_derivative: ', target_derivative)
                    # print('tiled_pitches: ',tiled_pitches)
                    resulted_derivative = tiled_pitches - pitch_grid[-1]
                    #resulted_derivative_in_degree = np.array(
                    #    [pc_motion_to_scale_degree_motion(x, tiled_scale_pc_distribution_grid, pitch_grid[-1]) for x in
                    #     resulted_derivative])
                    resulted_direction = 1 * (resulted_derivative > 0) + 0 * (np.abs(resulted_derivative) == 0) + -1 * (
                                resulted_derivative < 0)
                    # print('target_derivative,resulted_derivative: ',target_derivative,resulted_derivative)
                    derivative_diff = np.abs(target_derivative - resulted_derivative)
                    direction_diff = np.abs(target_direction - resulted_direction)
                    # print('derivative_diff:',derivative_diff)


                else:
                    derivative_diff = np.zeros(tiled_pc_distribution_grid.size, dtype=int)
                    direction_diff = np.zeros(tiled_pc_distribution_grid.size, dtype=int)
                force = (tiled_scale_pc_distribution_grid)*(10*tiled_chord_pc_distribution_grid+ 0.01 * (1 / (1 + np.abs(diff))) + 1 * (
                                1 / (1 + 1 * derivative_diff)) + 1 * (1 / (1 + direction_diff)))  # shape = (3*12,1)
                assert np.all(1 - np.isnan(force))
                argmax = np.argmax(force)  # shape = (,)
                if argmax == 0:
                    print('argmax == 0, force == ' + str(force))

                # print('argmax.shape:', argmax.shape)
                # print('argmax%12: ',argmax%12)
                pitch = tiled_pitches[argmax]
                # print('{:>5} {:>5}'.format(*['derivative_diff[argmax]',str(derivative_diff[argmax])]
                #                                       #+ ['target_derivative',str(target_derivative)]
                #                                       ))
            elif x == '-':
                pitch = pitch_grid[i - 1]
            else:
                pitch = 999

            pitch_grid.append(pitch)
        # Plot.draw_contour_minimal(approx_contour)
        return pitch_grid


class StreamBuilder:
    @staticmethod
    def measures_to_stream(measures: list[(object, object, object, object)], ignore_offset=False, title='original'):

        stream = m21.stream.Stream()
        stream.append(m21.metadata.Metadata(title=title))
        stream.append(m21.meter.TimeSignature('3/4'))
        key = measures[0][1]
        Mm = key.mode

        for i, test_measure in enumerate(measures):
            m21_measure = copy.deepcopy(test_measure[0])
            stream.append(m21_measure)

        stream.show()

        # sys.exit()

    @staticmethod
    def pitch_grid_rhythm_grid_to_stream(pitch_grid: list, rhythm_grid: list,
                                         key: m21.key.Key, text=None) -> m21.stream.Measure:
        stream = m21.stream.Measure()
        stream.append(m21.key.KeySignature(key.sharps))
        stream.duration = m21.duration.Duration(quarterLength=3)
        note_list = []
        for i, x in enumerate(rhythm_grid):

            if x == 'x':
                note = m21.note.Note(key.tonic.midi + pitch_grid[i], quarterLength=0.25)
                note_list.append(note)
            elif x == '_':
                note = m21.note.Rest(quarterLength=0.25)
                note_list.append(note)
            elif x == '-':
                note_list[-1].duration.quarterLength += 0.25
        note_list[0].lyric = text
        for note in note_list:
            stream.append(note)
        return stream


class FeatureSequenceEvaluation:
    def __init__(self):
        pass

    def pitch_continuity(self, ):
        return

    def rhythm_continuity(self, ):
        return

class PieceGeneration:
    #random_features, feature_distribution, feature_evaluator = construct_sample_space_and_evaluator(pieces)
    #list_of_latent_features, target_locations = construct_templates(feature_distribution)
    #fits = evaluate_fits(features=random_features, latent_features=list_of_latent_features)
    #assemble_piece(random_features, fits, target_locations=target_locations)
    def __init__(self):
        # set up variables, need to be specified
        self.pieces = None
        self.search_space_constructor = None
        self.templates_constructor = None
        self.evaluator = None
        self.piece_assembler = None
        # intermediate variables
        self.search_space = None
        self.templates = None
        self.fits = None
        # final_results
        self.generated_pieces = None

    def generate(self):
        self.search_space = self.search_space_constructor(pieces=self.pieces)
        self.templates = self.templates_constructor(pieces=self.pieces)
        self.fits = self.evaluator(search_space=self.search_space, templates=self.templates)
        self.generated_pieces = self.piece_assembler(search_space = self.search_space, fits=self.fits, templates = self.templates)


class Experiment:
    mode_pc_onehot = {
        'major': np.tile([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], (12, 1)),
        'minor': np.tile([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], (12, 1)),
        'dorian': np.tile([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], (12, 1)),
        'chromatic': np.tile([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], (12, 1))
    }

    chord_pc_onehot = {
        'I': np.tile([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], (12, 1)),
        'ii': np.tile([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], (12, 1)),
        'iii': np.tile([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], (12, 1)),
        'IV': np.tile([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], (12, 1)),
        'V': np.tile([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], (12, 1)),
        'vi': np.tile([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], (12, 1)),
        'vii': np.tile([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], (12, 1)),

        'i': np.tile([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], (12, 1)),
        'iio': np.tile([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], (12, 1)),
        'III': np.tile([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], (12, 1)),
        'iv': np.tile([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], (12, 1)),
        # 'V': np.tile([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], (12, 1)),
        'VI': np.tile([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], (12, 1)),
        # 'vii': np.tile([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], (12, 1)),
    }

    def __init__(self):
        xml_files = ['../data/xml/' + file_name for file_name in
                     filter(lambda x: x.endswith('.xml'), sorted(os.listdir('../data/xml')))]

        samples = random.sample(xml_files, k=599)

        def get_specific_pieces():
            specific_sample = ['../data/xml/Polonäs_6b9196.xml',
                               '../data/xml/_Polska_efter_Petter_Dufva_75c046.xml',
                               '../data/xml/_Polonäs_sexdregasamlingen_del_2_nr_40_f6ee90.xml',
                               '../data/xml/Polonäs_i_Dm_efter_Daniel_Danielsson_ac3754.xml',
                               '../data/xml/Polonäs_ab479a.xml',
                               '../data/xml/Slängpolska_från_Torna_Hällestad_d2f7e4.xml',
                               '../data/xml/Pollonesse_183229_d6da88.xml',
                               '../data/xml/Polonäs_efter_Pehr_Andersson_Bild_20_nr_56_20b592.xml',
                               '../data/xml/_Fingertarmen_polska_efter_ByssKalle_nr_46_b824bf.xml',
                               '../data/xml/Pollonesse_183212_de27a4.xml',
                               '../data/xml/Polonäs_sexdregasamlingen_del_3_nr_27_ff1c20.xml',
                               '../data/xml/Polska_efter_Ida_i_Rye_4b4d84.xml',
                               '../data/xml/Polonäs_av_Forssén_f59468.xml',
                               ]

            specific_pieces = []

            for xml_file in specific_sample:
                test_piece = m21.converter.parse(xml_file)
                voice = test_piece.getElementsByClass(m21.stream.Part)[0]
                measures = voice.getElementsByClass(m21.stream.Measure)
                duration = measures[0].duration.quarterLength
                key_signature = test_piece.flat.getKeySignatures()[0]
                time_signature = test_piece.flat.getTimeSignatures()[0]
                measures_with_piece_info = [(m, key_signature, time_signature, xml_file) for m in measures]
                specific_pieces.append(measures_with_piece_info)
            return specific_pieces

        def get_harmony_fitting_watchlist_pieces():
            harmony_fitting_watchlist = ['../data/xml/Polonäs_fd6cd9.xml',
                                         '../data/xml/Polonäs_av_Forssén_f59468.xml',
                                         '../data/xml/Polonäs_ab479a.xml',
                                         ]
            harmony_fitting_watchlist_pieces = []
            for xml_file in harmony_fitting_watchlist:
                test_piece = m21.converter.parse(xml_file)
                voice = test_piece.getElementsByClass(m21.stream.Part)[0]
                measures = voice.getElementsByClass(m21.stream.Measure)
                duration = measures[0].duration.quarterLength
                key_signature = test_piece.flat.getKeySignatures()[0]
                time_signature = test_piece.flat.getTimeSignatures()[0]
                measures_with_piece_info = [(m, key_signature, time_signature, xml_file) for m in measures]
                harmony_fitting_watchlist_pieces.append(measures_with_piece_info)
            return harmony_fitting_watchlist_pieces

        def get_D_major_pieces(samples):
            D_major_pieces = []
            for xml_file in samples:
                test_piece = m21.converter.parse(xml_file)
                voice = test_piece.getElementsByClass(m21.stream.Part)[0]
                measures = voice.getElementsByClass(m21.stream.Measure)
                key_signature = test_piece.flat.getKeySignatures()[0]
                if key_signature == m21.key.Key('D'):
                    time_signature = test_piece.flat.getTimeSignatures()[0]
                    measures_with_piece_info = [(m, key_signature, time_signature, xml_file) for m in measures]
                    if len(measures_with_piece_info) > 1:
                        D_major_pieces.append(measures_with_piece_info)
                    if len(measures_with_piece_info) == 8:
                        print(xml_file)
            return D_major_pieces

        def get_cadences(pieces):
            all_measures = sum(pieces, [])
            eight_bar_pieces = [x for x in pieces if len(x) in [8]]
            print('len(eight_bar_pieces): ', len(eight_bar_pieces))
            assert len(eight_bar_pieces) > 0

            def measure_contain_end_repeat(m: m21.stream.Measure) -> bool:
                repeat_marks = list(m.getElementsByClass(m21.repeat.RepeatMark))
                repeat_marks = [x for x in repeat_marks if hasattr(x, 'direction')]
                end_repeat_marks = [x for x in repeat_marks if x.direction == 'end']
                return end_repeat_marks != []

            measure_with_repeat_end = [x for x in all_measures if measure_contain_end_repeat(x[0])]
            cadential_measures = list(
                set([x[-1] for x in pieces] + [x[3] for x in eight_bar_pieces] + measure_with_repeat_end))
            ending_measures = list(set([x[-1] for x in pieces]))

            print('len(measure_with_repeat_end): ', len(measure_with_repeat_end))
            print('len(cadential_measures): ', len(cadential_measures))
            strong_candeces = [x for x in cadential_measures if
                               BarPatternFeatures.contour_sixteenth_grid(x[0], x[1], x[2])[
                                   -1] % 12 == 0] + ending_measures
            strong_candeces = list(set(strong_candeces))
            weak_cadences = list(set([x for x in cadential_measures if x not in strong_candeces]))
            return strong_candeces, weak_cadences

        #self.specific_pieces = get_specific_pieces()
        #self.D_major_pieces = get_D_major_pieces(samples=samples)
        #self.strong_candeces, self.weak_cadences = get_cadences(pieces=self.D_major_pieces)
        self.harmony_fitting_watchlist_pieces = get_harmony_fitting_watchlist_pieces()

    @staticmethod
    def recover_measures(measures: list[m21.stream.Measure]):

        constructed_melodies = []
        stream = m21.stream.Stream()
        stream.append(m21.metadata.Metadata(title='original'))
        stream_constructed_melodies = m21.stream.Stream()
        stream_constructed_melodies.append(m21.metadata.Metadata(title='reconstructed'))
        stream_constructed_melodies.append(m21.meter.TimeSignature('3/4'))
        key = measures[0][1]
        stream_constructed_melodies.append(key)
        Mm = key.mode
        Mm = 'chromatic'
        print('Mm: ', Mm)

        pc_distribution_grid = Experiment.mode_pc_onehot[Mm]
        for test_measure in measures:
            m21_measure = test_measure[0]
            stream.append(m21_measure)
            if m21_measure.duration.quarterLength >= 2.0:
                test_contour = BarPatternFeatures.contour_cosine(test_measure[0], key=key,
                                                                 n_beat=test_measure[2].numerator)
                test_rhythm = BarPatternFeatures.rhythm_sixteenth_grid(test_measure[0], key=key,
                                                                       n_beat=test_measure[2].numerator)
                constructed_melody = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=test_contour,
                                                                               rhythm_grid=test_rhythm, vl=None,
                                                                               sample_point_size=300,
                                                                               pc_distribution_grid=pc_distribution_grid)
                constructed_stream = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody, test_rhythm,
                                                                                    key=key)

                constructed_melodies.append(constructed_melody)
                stream_constructed_melodies.append(constructed_stream)

        stream.show()
        stream_constructed_melodies.show()
        # sys.exit()

    @staticmethod
    def recover_measures_with_diffrent_pc_distribution(
            measures: list[(m21.stream.Measure, m21.key.Key, m21.meter.TimeSignature, str)]):
        print(measures)
        constructed_melodies = []
        stream = m21.stream.Stream()
        stream.append(m21.metadata.Metadata(title=measures[0][-1]))
        stream_constructed_melodies = m21.stream.Stream()
        stream_constructed_melodies.append(m21.metadata.Metadata(title='Enforce harmony'))
        stream_constructed_melodies.append(m21.meter.TimeSignature('3/4'))
        key = measures[0][1]
        stream_constructed_melodies.append(key)
        Mm = key.mode

        pc_distribution_grid = Experiment.mode_pc_onehot[Mm]
        piece_pc_distribution_grid = np.tile(pc_distribution_grid, (len(measures), 1, 1))
        chord_sequence_for_M = ['I', 'ii', 'V', 'I']
        chord_sequence_for_m = ['i', 'iio', 'V', 'VI', 'i', 'iv', 'V', 'i']
        chord_sequence_for_dorian = ['i', 'ii', 'V', 'i', 'i', 'IV', 'V', 'i']
        if Mm == 'major':
            chord_sequence = chord_sequence_for_M
        if Mm == 'minor':
            chord_sequence = chord_sequence_for_m
        if Mm == 'dorian':
            chord_sequence = chord_sequence_for_dorian
        else:
            print(Mm)

        chord_preference_grid = np.array([Experiment.chord_pc_onehot[x] for x in chord_sequence])
        piece_pc_distribution_grid = piece_pc_distribution_grid[len(chord_preference_grid)] ## same length as template
        piece_pc_distribution_grid = piece_pc_distribution_grid + 1 * chord_preference_grid
        if len(measures) == 1:
            measures = measures * len(piece_pc_distribution_grid)
        for i, (measure, pc_distribution_grid, chord) in enumerate(
                zip(measures, piece_pc_distribution_grid, chord_sequence)):

            # print(i, (measure,pc_distribution_grid))
            if measure[0] not in stream:
                stream.append(measure[0])
            test_contour = BarPatternFeatures.contour_cosine(measure[0], key=key,
                                                             n_beat=measure[2].numerator)
            test_rhythm = BarPatternFeatures.rhythm_sixteenth_grid(measure[0], key=key,
                                                                   n_beat=measure[2].numerator)
            constructed_melody = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=test_contour,
                                                                           rhythm_grid=test_rhythm, vl=None,
                                                                           sample_point_size=300,
                                                                           pc_distribution_grid=
                                                                           piece_pc_distribution_grid[i])

            constructed_stream = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody, test_rhythm,
                                                                                key=key, text=chord)

            # print(test_rhythm)
            constructed_melodies.append(constructed_melody)
            stream_constructed_melodies.append(constructed_stream)

        # print(constructed_melodies)
        stream.show()
        stream_constructed_melodies.show()
        # sys.exit()

    @staticmethod
    def mix_match_rhythm_contour(
            first_piece_measures: list[(m21.stream.Measure, m21.key.KeySignature, m21.meter.TimeSignature)],
            second_piece_measures: list[(m21.stream.Measure, m21.key.KeySignature, m21.meter.TimeSignature)]):

        constructed_melodies = []
        stream1 = m21.stream.Stream()
        stream1.append(m21.metadata.Metadata(title='Original 1 (provide contour and key)'))
        stream2 = m21.stream.Stream()
        stream2.append(m21.metadata.Metadata(title='Original 2 (provide rhythm)'))
        stream_constructed_melodies1 = m21.stream.Stream()
        stream_constructed_melodies1.append(m21.metadata.Metadata(title='Reconstructed 1'))
        stream_constructed_melodies1.append(m21.meter.TimeSignature('3/4'))
        stream_constructed_melodies2 = m21.stream.Stream()
        stream_constructed_melodies2.append(m21.metadata.Metadata(title='Reconstructed 2'))
        stream_constructed_melodies2.append(m21.meter.TimeSignature('3/4'))

        key1 = first_piece_measures[0][1]
        key2 = second_piece_measures[0][1]

        Mm1 = key1.mode
        Mm2 = key2.mode
        print('Mm: ', Mm1, Mm2)
        pc_distribution_grid1 = Experiment.mode_pc_onehot[Mm1]
        pc_distribution_grid2 = Experiment.mode_pc_onehot[Mm2]

        stream_constructed_melodies1.append(key1)
        stream_constructed_melodies2.append(key2)
        for first_piece_measure, second_piece_measure in zip(first_piece_measures, second_piece_measures):
            stream1.append(first_piece_measure[0])
            stream2.append(second_piece_measure[0])

            contour1 = BarPatternFeatures.contour_cosine(first_piece_measure[0], key=key1,
                                                         n_beat=first_piece_measure[2].numerator)

            rhythm1 = BarPatternFeatures.rhythm_sixteenth_grid(first_piece_measure[0], key=key1,
                                                               n_beat=first_piece_measure[2].numerator)

            contour2 = BarPatternFeatures.contour_cosine(second_piece_measure[0], key=key2,
                                                         n_beat=second_piece_measure[2].numerator)
            rhythm2 = BarPatternFeatures.rhythm_sixteenth_grid(second_piece_measure[0], key=key2,
                                                               n_beat=second_piece_measure[2].numerator)

            constructed_melody1 = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=contour1,
                                                                            rhythm_grid=rhythm2,
                                                                            vl=None,
                                                                            sample_point_size=300,
                                                                            pc_distribution_grid=pc_distribution_grid1)
            constructed_melody2 = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=contour2,
                                                                            rhythm_grid=rhythm1,
                                                                            vl=None,
                                                                            sample_point_size=300,
                                                                            pc_distribution_grid=pc_distribution_grid2)

            rhythm_intersection = Combination.rhythm_intersection(rhythm1, rhythm2)
            rhythm_union = Combination.rhythm_union(rhythm1, rhythm2)

            constructed_stream1 = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody1, rhythm_union,
                                                                                 key=key1)

            constructed_stream2 = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody2, rhythm_union,
                                                                                 key=key2)

            # print(constructed_melody)
            # print(rhythm)
            constructed_melodies.append(constructed_melody1)
            stream_constructed_melodies1.append(constructed_stream1)
            stream_constructed_melodies2.append(constructed_stream2)

        # print(constructed_melodies)
        stream1.show()
        stream2.show()
        stream_constructed_melodies1.show()
        stream_constructed_melodies2.show()
        sys.exit()

    @staticmethod
    def recovering_with_varying_contour_hierarchy(
            piece_measures: list[(m21.stream.Measure, m21.key.KeySignature, m21.meter.TimeSignature)]):
        stream = m21.stream.Stream()
        stream.append(m21.metadata.Metadata(title='original'))
        for measure in piece_measures:
            stream.append(measure[0])

        for i, level in enumerate(['4', '8', '16']):
            constructed_melodies = []
            stream_constructed_melodies = m21.stream.Stream()
            stream_constructed_melodies.append(m21.metadata.Metadata(title='reconstructed level = ' + level))
            stream_constructed_melodies.append(m21.meter.TimeSignature('3/4'))
            key = piece_measures[0][1]
            stream_constructed_melodies.append(key)
            Mm = key.mode
            pc_distribution_grid = Experiment.mode_pc_onehot[Mm]
            for measure in piece_measures:
                pitches, contours, coeffs = BarPatternFeatures.texture_contour_diff(measure)
                contours = np.array(contours)
                contour = np.sum(contours[0:i + 1], axis=0)
                coeff = np.sum(coeffs[0:i + 1], axis=0)
                # print('this_level_contour.shape: ',contour.shape)
                rhythms = BarPatternFeatures.rhythm_incremental(measure[0], key=key,
                                                                n_beat=measure[2].numerator)
                rhythm = rhythms[i]

                constructed_melody = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=coeff,
                                                                               rhythm_grid=rhythm, vl=None,
                                                                               sample_point_size=300,
                                                                               pc_distribution_grid=pc_distribution_grid,
                                                                               type='coeff')
                # print(constructed_melody)
                constructed_stream = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody, rhythm,
                                                                                    key=key)
                constructed_melodies.append(constructed_melody)
                stream_constructed_melodies.append(constructed_stream)

            stream_constructed_melodies.show()
        stream.show()
        sys.exit()

    @staticmethod
    def mix_match_contour_surface(
            first_piece_measures: list[(m21.stream.Measure, m21.key.KeySignature, m21.meter.TimeSignature)],
            second_piece_measures: list[(m21.stream.Measure, m21.key.KeySignature, m21.meter.TimeSignature)]):
        constructed_melodies = []
        stream1 = m21.stream.Stream()
        stream1.append(m21.metadata.Metadata(title='Original 1 (provide contour and key)'))
        stream2 = m21.stream.Stream()
        stream2.append(m21.metadata.Metadata(title='Original 2 (provide rhythm)'))
        stream_constructed_melodies1 = m21.stream.Stream()
        stream_constructed_melodies1.append(m21.metadata.Metadata(title='Reconstructed 1'))
        stream_constructed_melodies1.append(m21.meter.TimeSignature('3/4'))
        stream_constructed_melodies2 = m21.stream.Stream()
        stream_constructed_melodies2.append(m21.metadata.Metadata(title='Reconstructed 2'))
        stream_constructed_melodies2.append(m21.meter.TimeSignature('3/4'))

        key1 = first_piece_measures[0][1]
        key2 = second_piece_measures[0][1]

        Mm1 = key1.mode
        Mm2 = key2.mode
        # Mm1 = 'chromatic'
        # Mm2 = 'chromatic'
        print('Mm: ', Mm1, Mm2)
        pc_distribution_grid1 = Experiment.mode_pc_onehot[Mm1]
        pc_distribution_grid2 = Experiment.mode_pc_onehot[Mm2]

        stream_constructed_melodies1.append(key1)
        stream_constructed_melodies2.append(key2)

        for first_piece_measure, second_piece_measure in zip(first_piece_measures, second_piece_measures):
            stream1.append(first_piece_measure[0])
            stream2.append(second_piece_measure[0])

            pitches1, contours1, coeff1 = BarPatternFeatures.texture_contour_diff(first_piece_measure)
            pitches2, contours2, coeff2 = BarPatternFeatures.texture_contour_diff(second_piece_measure)

            rhythms1 = BarPatternFeatures.rhythm_incremental(first_piece_measure[0], key=key2,
                                                             n_beat=first_piece_measure[2].numerator)
            rhythms2 = BarPatternFeatures.rhythm_incremental(second_piece_measure[0], key=key2,
                                                             n_beat=second_piece_measure[2].numerator)

            coeff1 = coeff1[0] + coeff1[1] + coeff1[2]
            coeff2 = coeff2[0] + coeff2[1] + coeff2[2]
            boundary_index = int(coeff1.shape[0] * 2 / 3)
            print('coeff1.shape: ', coeff1.shape, 'boundary_index: ', boundary_index)
            new_ceoff1 = np.concatenate([coeff1[:boundary_index], coeff2[boundary_index:]])
            new_ceoff2 = np.concatenate([coeff2[:boundary_index], coeff1[boundary_index:]])

            rhythm1 = rhythms2[2]
            rhythm2 = rhythms1[2]

            constructed_melody1 = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=new_ceoff1,
                                                                            rhythm_grid=rhythm2,
                                                                            vl=None,
                                                                            sample_point_size=300,
                                                                            pc_distribution_grid=pc_distribution_grid1,
                                                                            type='coeff')
            constructed_melody2 = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=new_ceoff2,
                                                                            rhythm_grid=rhythm1,
                                                                            vl=None,
                                                                            sample_point_size=300,
                                                                            pc_distribution_grid=pc_distribution_grid2,
                                                                            type='coeff')

            constructed_stream1 = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody1, rhythm2, key=key1)

            constructed_stream2 = StreamBuilder.pitch_grid_rhythm_grid_to_stream(constructed_melody2, rhythm1, key=key2)

            # print(constructed_melody)
            # print(rhythm)
            constructed_melodies.append(constructed_melody1)
            stream_constructed_melodies1.append(constructed_stream1)
            stream_constructed_melodies2.append(constructed_stream2)

        # print(constructed_melodies)
        stream1.show()
        stream2.show()
        stream_constructed_melodies1.show()
        stream_constructed_melodies2.show()
        sys.exit()

    @staticmethod
    def clustering_contour_diff(all_measures):
        all_pitches = [BarPatternFeatures.contour_sixteenth_grid(measure[0], key=measure[1],
                                                                 n_beat=measure[2].numerator) for measure in
                       all_measures]
        _, __, coeffs = BarPatternFeatures.batch_texture_contour_diff(all_measures)
        print('full coeffs.shape: ', coeffs.shape)
        coeffs = coeffs[:, :, :]
        # cumulative_coeffs = np.array([np.sum(coeffs[:, :i + 1, :], axis=1) for i in range(3)]).transpose((1, 0, 2))
        # coeffs = cumulative_coeffs
        print('coeffs.shape: ', coeffs.shape)

        dct_coeffs = np.array([BarPatternFeatures.contour_cosine(measure[0], key=measure[1],
                                                                 n_beat=measure[2].numerator) for measure in
                               all_measures])
        weighted_dct_coeffs = np.arange(dct_coeffs.shape[1])[::-1] * dct_coeffs
        print(weighted_dct_coeffs)
        print('calculating distance')

        diff = np.expand_dims(coeffs, axis=1) - np.expand_dims(coeffs, axis=0)
        _D = np.linalg.norm(diff, axis=-1, ord=2)
        means = np.average(_D, axis=(0, 1))
        print('means in 3 hierarchy distance:', means)
        # _D = _D/means
        print('_D.shape: ', _D.shape)
        metrical_weight = np.power(0.1, np.arange(3))
        print('_D: ', _D)
        weighted__D = _D * metrical_weight
        print('weighted__D: ', weighted__D)
        D = np.linalg.norm(weighted__D, axis=-1, ord=2)

        print('D.shape: ', D.shape)
        print('unique distance percentage:',
              2 * np.unique(D).shape[0] / (D.size - D.shape[0]))

        D_dct = np.linalg.norm(
            np.expand_dims(weighted_dct_coeffs, axis=1) - np.expand_dims(weighted_dct_coeffs, axis=0), axis=-1, ord=2)
        model = cluster.AgglomerativeClustering(n_clusters=100, affinity='precomputed',
                                                linkage='average')
        model_dct = cluster.AgglomerativeClustering(n_clusters=100, affinity='precomputed',
                                                    linkage='average')

        model.fit(D)
        model_dct.fit(D_dct)
        piano_roll = np.arange(-12 * 3, 12 * 3)
        piano_roll = piano_roll.reshape((1, 1, -1))
        all_pitches = np.array(all_pitches)[..., np.newaxis]
        onehot = all_pitches == piano_roll
        onehot = np.array(onehot, dtype=int)
        onehot = np.array(onehot, dtype=str)
        onehot = np.flip(onehot, axis=-1)
        onehot = np.transpose(onehot, (0, 2, 1))
        onehot_list = onehot.tolist()

        def replace_string(list_of_string):
            new_list = []
            for x in list_of_string:
                if x == '0':
                    new_list.append('. . . . . ')
                if x == '1':
                    new_list.append('<===>')
            return new_list

        onehot_str = [
            ''.join(
                [
                    (12 * '{:2s}' + '<br>').format(
                        *replace_string(x)
                    )
                    for x in y
                ]
            )
            for y in onehot_list]

        Plot.draw_scatter(distance_matrix=[D, D_dct], text_arrays=onehot_str, file_name='contour metrical',
                          model=[model, model_dct])
        sys.exit()

    @staticmethod
    def clustering_legacy(all_measures):
        sample_point_size = 120
        measure_feature = np.array([
            BarPatternFeatures.contour_cosine(measure[0], key=measure[1],
                                              n_beat=measure[2].numerator) for
            measure in all_measures])
        contour_arrays_text = np.array([
            BarPatternFeatures.contour_sixteenth_grid(measure[0], key=measure[1],
                                                      n_beat=measure[2].numerator, n_grid=12) for
            measure in all_measures])
        contour_arrays = np.array([
            BarPatternFeatures.contour_sixteenth_grid(measure[0], key=measure[1],
                                                      n_beat=measure[2].numerator, n_grid=120) for
            measure in all_measures])

        print('calculating distance')
        _Levenshtein = distance.pdist(measure_feature, )  # DistanceFunction.rhythm_grid_distance)
        Levenshtein = distance.squareform(_Levenshtein)
        print(Levenshtein)
        print('unique distance percentage:',
              2 * np.unique(Levenshtein).shape[0] / (Levenshtein.size - Levenshtein.shape[0]))
        print('calculating AgglomerativeClustering')
        model = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=3, affinity='precomputed',
                                                linkage='average')
        model.fit(Levenshtein)
        linkage_matrix = linkage(_Levenshtein)
        text_array = list(zip(contour_arrays_text, measure_feature))
        contour_approxs = [[scipy.fftpack.idct(x[:i + 1], norm='backward', n=sample_point_size) for i in range(len(x))]
                           for
                           x in measure_feature]
        print(measure_feature)
        Plot.draw_scatter(distance_matrix=Levenshtein, text_arrays=text_array, file_name='contour')

    @staticmethod
    def generate_measure(measure, pc_distribution_grid: np.ndarray = None, contour_coeff=None):
        import piece
        print('measure: ', measure)
        tree = piece.Tree()
        piece.node_grow(tree=tree, level='Top')
        piece.node_grow(tree=tree, level='Section')
        piece.node_grow(tree=tree, level='Phrase')
        piece.node_grow(tree=tree, level='Subphrase')
        piece.node_grow(tree=tree, level='Bar')
        piece.node_grow(tree=tree, level='Beat')
        test_bar = tree.children[0].children[0].children[0].children[0]
        # print(test_bar)
        contour_coeff = BarPatternFeatures.contour_cosine(measure[0], measure[1], n_beat=measure[2])
        pc_grid = test_bar.get_pc_grid()
        # print('pc_grid: ',pc_grid)

        rhythm_grid = test_bar.get_rhythm_grid()
        # print('rhythm_grid: ', rhythm_grid)
        pc_distribution_grid = None
        pitch_grid = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=contour_coeff, rhythm_grid=rhythm_grid,
                                                               pc_distribution_grid=pc_grid, sample_point_size=300,
                                                               vl=None, type='coeff')
        stream = StreamBuilder.pitch_grid_rhythm_grid_to_stream(pitch_grid, rhythm_grid, key=measure[1])
        stream.show()

    @staticmethod
    def generate_piece_from_tree(contour_coeff_dict):
        import piece
        tree = piece.Tree()
        piece.node_grow(tree=tree, level='Top')
        piece.node_grow(tree=tree, level='Section')
        piece.node_grow(tree=tree, level='Phrase')
        piece.node_grow(tree=tree, level='Subphrase')
        piece.node_grow(tree=tree, level='Bar')
        piece.node_grow(tree=tree, level='Beat')
        from tree_to_xml import tree_to_stream_powerful
        tree_to_stream_powerful(tree).show()
        bars = [bar for section in tree.children for phrase in section.children for subphrase in phrase.children for bar
                in subphrase.children]

        stream = m21.stream.Stream()
        stream.append(m21.meter.TimeSignature('3/4'))
        key_of_tree = tree.data.key.symbols[0].char

        key_of_tree = m21.key.Key(key_of_tree)
        scale_M = Experiment.mode_pc_onehot['major'][0]

        form_contour_dict = dict()

        opening_phrase_contour = contour_coeff_dict['opening'][:2]
        ending_phrase_contour = contour_coeff_dict['ending'][:2]
        middle_phrase_contour = contour_coeff_dict['middle'][:2]

        def rotate(l, n):
            return l[n:] + l[:n]

        for i, bar in enumerate(bars):
            current_form = bar.data.form.symbols[0].char
            if i < 4:
                contour_coeff = opening_phrase_contour[0]
                opening_phrase_contour = rotate(opening_phrase_contour, 1)
            elif i >= len(bars) - 2:
                contour_coeff = ending_phrase_contour[0]
                ending_phrase_contour = rotate(ending_phrase_contour, 1)
            else:
                contour_coeff = middle_phrase_contour[0]
                middle_phrase_contour = rotate(middle_phrase_contour, 1)
            form_contour_dict[current_form] = contour_coeff

            rhythm_grid = bar.get_rhythm_grid()
            # print('rhythm_grid: ', rhythm_grid)
            final_pc_grid = bar.get_pc_grid() + scale_M

            # print('final_pc_grid: ',final_pc_grid)

            pitch_grid = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=contour_coeff,
                                                                   rhythm_grid=rhythm_grid,
                                                                   pc_distribution_grid=final_pc_grid,
                                                                   sample_point_size=300,
                                                                   vl=None, type='coeff')

            constructed_stream = StreamBuilder.pitch_grid_rhythm_grid_to_stream(pitch_grid, rhythm_grid,
                                                                                key=key_of_tree)
            stream.append(constructed_stream)

        stream.show()

    @staticmethod
    def contour_self_similarity(measures):
        StreamBuilder.measures_to_stream(measures, title=measures[0][-1])
        n = len(measures)
        contours = np.array(
            [BarPatternFeatures.contour_cosine(measure=x[0], key=x[1], n_beat=x[2])[1:] for x in measures])

        scale_degree_and_onset_and_duration = np.array(
            [BarPatternFeatures.scale_degree_and_onset_and_duration(measure=x[0], key=x[1], n_beat=x[2]) for x in
             measures])
        _levenstein_distance = scipy.spatial.distance.pdist(scale_degree_and_onset_and_duration.reshape(-1, 1),
                                                            DistanceFunction.levenshtein_with_operation_cost)
        levenstein_distance = scipy.spatial.distance.squareform(_levenstein_distance)
        diff = np.expand_dims(contours, axis=0) - np.expand_dims(contours, axis=1)
        distance = np.linalg.norm(diff, ord=2, axis=-1)

        pitches = np.array([BarPatternFeatures.texture_contour_diff(measure=x)[0] for x in measures])
        pc_dist = pitches.reshape(pitches.shape[0], -1, 1)
        pc_dist = np.sum(np.equal(pc_dist, np.arange(24)), axis=1)
        pc_dist = pc_dist[:, :12] + pc_dist[:, 12:]
        print(pc_dist.shape)
        pc_dist = np.array(pc_dist > 0, dtype=int)

        dictance_pc_dist = np.linalg.norm(np.expand_dims(pc_dist, axis=0) - np.expand_dims(pc_dist, axis=1), ord=2,
                                          axis=-1)

        fig = plt.figure()
        gs = plt.GridSpec(2, 4, figure=fig)

        contour_D_matrix = 1 - distance / np.max(distance)
        levenstein_D_matrix = 1 - levenstein_distance / np.max(levenstein_distance)
        pc_dict_D_matrix = 1 - dictance_pc_dist / np.max(dictance_pc_dist)

        alpha = 0.9
        background_matrix = np.zeros(shape=(n, n))
        levenstein_D_matrix = np.stack([levenstein_D_matrix, background_matrix, background_matrix], axis=-1)
        ax_levenstein_D = fig.add_subplot(gs[0, 0])
        ax_levenstein_D.set_title('(degree,onset,duration) self-similarity')
        ax_levenstein_D.set_xticks(np.arange(1, n + 1))
        ax_levenstein_D.set_yticks(np.arange(1, n + 1))
        ax_levenstein_D.imshow(levenstein_D_matrix, extent=[1, n + 1, n + 1, 1], alpha=alpha)

        contour_D_matrix = np.stack([background_matrix, contour_D_matrix, background_matrix], axis=-1)
        ax_contour_d = fig.add_subplot(gs[0, 1])
        ax_contour_d.set_title('contour (bar) self-similarity')
        ax_contour_d.set_xticks(np.arange(1, n + 1))
        ax_contour_d.set_yticks(np.arange(1, n + 1))
        ax_contour_d.imshow(contour_D_matrix, extent=[1, n + 1, n + 1, 1], alpha=alpha)

        pc_dict_D_matrix = np.stack([background_matrix, background_matrix, pc_dict_D_matrix], axis=-1)
        ax_pc_dist = fig.add_subplot(gs[0, 2])
        ax_pc_dist.set_title('pc distribution self-similarity')
        ax_pc_dist.set_xticks(np.arange(1, n + 1))
        ax_pc_dist.set_yticks(np.arange(1, n + 1))
        ax_pc_dist.imshow(pc_dict_D_matrix, extent=[1, n + 1, n + 1, 1], alpha=alpha)

        ax_combined_D = fig.add_subplot(gs[0, 3])
        ax_combined_D.set_title('combined self-similarity')
        # ax_combined_D.imshow(np.maximum(contour_D_matrix,levenstein_D_matrix,pc_dict_D_matrix),extent=[1,n,n,1])
        ax_combined_D.set_xticks(np.arange(1, n + 1))
        ax_combined_D.set_yticks(np.arange(1, n + 1))
        sumed_matrix = contour_D_matrix + levenstein_D_matrix + pc_dict_D_matrix
        ax_combined_D.imshow(sumed_matrix, extent=[1, n + 1, n + 1, 1])

        ax_pitches = fig.add_subplot(gs[1, :])
        ax_pitches.scatter(np.linspace(1, len(pitches), 300 * len(pitches)), pitches, marker='s', c='grey', s=5)
        ax_pitches.set_xticks(np.arange(n))
        plt.show()

    @staticmethod
    def cadence_detection_test(pieces):
        _feature_distribution = sum([PieceFeatures.get_contour_rhythm_dict(piece) for piece in pieces], [])
        _feature_evaluator = FeatureEvaluation(joint_distribution_of_features=_feature_distribution)
        for piece in pieces:
            cadence_locations = PieceFeatures.cadence_detector(piece, _feature_evaluator)
            piece_title = piece[0][-1]
            piece_title = piece_title.replace('../data/xml/', '')
            print(piece[0][-1], cadence_locations)
            StreamBuilder.measures_to_stream(piece, title=piece_title)


    def feature_evaluation_at_work(self,pieces):
        def construct_sample_space_and_evaluator(pieces):
            # configuring feature evaluator
            print('-------- Begin experiment --------')
            print('len(pieces): ', len(pieces))
            print('-------- Begin segmentation --------')
            _feature_distribution = sum([PieceFeatures.get_contour_rhythm_dict(piece) for piece in pieces], [])
            _feature_evaluator = FeatureEvaluation(joint_distribution_of_features=_feature_distribution)
            piece_features = PieceFeatures(cadential_features=PieceFeatures.get_contour_rhythm_dict(self.strong_candeces))
            feature_distribution = sum(
                [piece_features.get_location_contour_rhythm_dict(piece, _feature_evaluator) for piece in pieces], [])
            feature_evaluator = FeatureEvaluation(joint_distribution_of_features=feature_distribution)

            print('-------- Constructing search space (breaking down components) --------')
            contour_distribution = [
                {'contour': feature['contour'], 'source': feature['source'], 'location': feature['location']} for
                feature in
                feature_distribution]
            rhythm_distribution = [
                {'rhythm': feature['rhythm'], 'source': feature['source'], 'location': feature['location']} for feature
                in
                feature_distribution]
            location_distribution = [{'location': feature['location'], 'source': feature['source']} for feature in
                                     feature_distribution]

            # make unique
            contour_distribution = list({i['contour'].tobytes(): i for i in contour_distribution}.values())
            rhythm_distribution = list({i['rhythm'].tobytes(): i for i in rhythm_distribution}.values())
            location_distribution = list({str(i['location']): i for i in location_distribution}.values())

            print('len(contour_distribution): ', len(contour_distribution))
            print('len(rhythm_distribution): ', len(rhythm_distribution))
            print('len(location_distribution): ', len(location_distribution))

            # construct search space for features
            seen_combinations = [(y['contour'].tolist(), y['rhythm'].tolist()) for y in feature_distribution]

            print('len(seen_combinations): ', len(seen_combinations))
            # print(seen_combinations)

            unseen_combinations = [(x, y) for x, y in itertools.product(contour_distribution, rhythm_distribution) if
                                   (x['contour'].tolist(), y['rhythm'].tolist()) not in seen_combinations]
            print('len(unseen_combinations): ', len(unseen_combinations))

            print('-------- Constructing search space (cartesian product of components) --------')
            random_features = [{'contour': x['contour'], 'rhythm': y['rhythm'],
                                'source': [x['source'], x['location'], y['source'], y['location']]} for x, y in
                               unseen_combinations]

            random_features = list({str(i): i for i in random_features}.values())
            return random_features, feature_distribution, feature_evaluator

        random_features, feature_distribution, feature_evaluator = construct_sample_space_and_evaluator(pieces)

        def construct_templates(feature_distribution):
            # evaluating search space and find best candidate features

            beginning_bars = [x for x in feature_distribution if x['location'][0] == 0]
            list_of_latent_features = [beginning_bars[0], beginning_bars[0], beginning_bars[0], None] + [
                beginning_bars[3],
                beginning_bars[3], beginning_bars[3], beginning_bars[3],
                beginning_bars[0],
                beginning_bars[0], beginning_bars[0],
                None] + [beginning_bars[3],
                         beginning_bars[3],
                         beginning_bars[3],
                         None]
            target_locations = [[0, 4], [1, 4], [2, 4], [3, 4]] \
                               + [[0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8]] + [[0, 4], [1, 4],
                                                                                                     [2, 4],
                                                                                                     [3, 4]]
            return list_of_latent_features, target_locations

        list_of_latent_features, target_locations = construct_templates(feature_distribution)

        def evaluate_fits(features: list[dict], latent_features: list[dict]):
            print('-------- Calculating all fits --------')
            internal_match_fits = []
            latent_match_fits = []
            location_match_fits = []
            l = len(random_features)

            for i, feature in enumerate(features):
                internal_match = feature_evaluator.internal_match(features=feature)
                latent_match = [feature_evaluator._match_with_latent(features=feature, latent_features=latent_feature)
                                for latent_feature in latent_features]
                location_match = [
                    feature_evaluator.cadence_only_location_match(features=feature, current_location=location) for
                    location in target_locations]
                # latent_match = np.zeros(len(latent_features))+1
                # location_match = np.zeros(len(target_locations)) + 1
                print('\r', str(i) + '/' + str(l), end='')
                internal_match_fits.append(internal_match)
                latent_match_fits.append(latent_match)
                location_match_fits.append(location_match)

            print('len(internal_match_fits): ', len(internal_match_fits))
            print('len(latent_match_fits): ', len(latent_match_fits))
            print('len(location_match_fits): ', len(location_match_fits))
            fits = {'internal': internal_match_fits, 'latent': latent_match_fits, 'location': location_match_fits}
            print(' ', 'done')
            return fits

        fits = evaluate_fits(features=random_features, latent_features=list_of_latent_features)

        def assemble_piece(features, fits, target_locations):
            all_top_features = []
            print('-------- Selecting best features in each position --------')
            internal_match_fits, latent_match_fits, location_match_fits = fits['internal'], fits['latent'], fits[
                'location']
            for i, current_location in enumerate(target_locations):
                total_scores = []
                print('current_location: ', current_location)
                for random_feature, internal_match_fit, latent_match_fit, location_match_fit in zip(features,
                                                                                                    internal_match_fits,
                                                                                                    latent_match_fits,
                                                                                                    location_match_fits):
                    # location_fit = feature_evaluator.location_match(features=random_feature,
                    #                                               current_location=current_location)
                    # location_fit = 1
                    # internal_match_fit = feature_evaluator.internal_match(features=random_feature)
                    # internal_match_fit = 1
                    # latent_match_fit = feature_evaluator.match_with_latent(features=random_feature,latent_features= list_of_latent_features[i])
                    # latent_match_fit = 1
                    # total_score = (location_match_fit[i]) * (latent_match_fit[i] ** 3) * (internal_match_fit)
                    total_score = (location_match_fit[i]) * (latent_match_fit[i] ** 1) * (internal_match_fit)
                    total_scores.append(total_score)
                total_scores = np.array(total_scores)
                arg_sort = np.argsort(-total_scores)
                features = np.array(features).reshape(-1)
                top_features = features[arg_sort[:10]]
                top_features = random.choices(top_features, k=10)
                all_top_features.append(top_features)
                print('len(top_features): ', len(top_features))

                # sampling_probabilities = np.exp(total_scores) / np.sum(np.exp(total_scores))
                # sampled_features = np.random.choice(random_features,10,p=sampling_probabilities)
                # all_top_features.append(sampled_features)

            print('-------- Assembling pieces from candidates in each position--------')
            candidates = list(zip(*all_top_features))[:5]
            # varying_source_pieces = [[] for _ in range(5)]
            # for j,_ in enumerate(varying_source_pieces):
            #    seen_source = []
            #    print('')
            #    for i,top_features in enumerate(all_top_features):
            #
            #        filterd_top_features = [x for x in top_features if x['source'][0] not in seen_source]
            #        print(len(filterd_top_features),end='=>')
            #        assert filterd_top_features != []
            #        if i== 0:
            #            selected_feature = filterd_top_features[j]
            #        else:
            #            selected_feature = filterd_top_features[0]
            #        _.append(selected_feature)
            #        print(seen_source,selected_feature['source'][0])
            #        seen_source.append(selected_feature['source'][0])

            for i, candidate in enumerate(candidates):
                print('--------------')
                print('feature source for candidate {}'.format(i + 1))
                for feature in candidate:
                    print(feature['source'])

            print('-------- constructing music21 score --------')
            # generate melody based on  candidate features
            target_chords = ['I', 'V', 'V', 'I'] + ['I', 'IV', 'I', 'ii', 'V', 'I', 'V', 'I'] + ['I', 'V', 'V', 'I']

            for i, candidate in enumerate(candidates):

                stream = m21.stream.Stream()
                stream.append(m21.metadata.Metadata(title='Top {}'.format(i + 1)))
                stream.append(m21.meter.TimeSignature('3/4'))
                stream.append(m21.key.KeySignature(2))
                pitch_grids = [MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=feature['contour'],
                                                                         rhythm_grid=feature['rhythm'],
                                                                         sample_point_size=300, vl=None,
                                                                         pc_distribution_grid=
                                                                         Experiment.chord_pc_onehot[current_chord])
                               for current_chord,feature in zip(target_chords,candidate)]
                streams = [
                    StreamBuilder.pitch_grid_rhythm_grid_to_stream(pitch_grid=pitch_grid, rhythm_grid=feature['rhythm'],
                                                                   key=m21.key.Key('D'),text=current_chord) for current_chord,pitch_grid, feature in
                    zip(target_chords,pitch_grids, candidate)]
                stream.append(streams)
                stream.show()

        assemble_piece(random_features, fits, target_locations=target_locations)

    @staticmethod
    def demo_internal_match(pieces):
        pass



if __name__ == '__main__':
    experiment = Experiment()
    # Experiment.recover_measures(measures=specific_pieces[-1])
    # Experiment.mix_match_rhythm_contour(eight_bar_pieces[2], eight_bar_pieces[3])
    # Experiment.clustering_contour_diff(all_measures)
    # Experiment.recovering_with_varying_contour_hierarchy(eight_bar_pieces[2])
    # Experiment.mix_match_contour_surface(eight_bar_pieces[2], eight_bar_pieces[3])
    # Experiment.recover_measures_with_diffrent_pc_distribution(experiment.harmony_fitting_watchlist_pieces[0][0:2])
    # Experiment.generate_measure(eight_bar_pieces[0][0])

    # all_contours = [BarPatternFeatures.contour_cosine(measure=x[0],key=x[1],n_beat=x[2]) for x in all_measures]
    # ending_contours = [BarPatternFeatures.contour_cosine(measure=x[0],key=x[1],n_beat=x[2]) for y in pieces for x in y[-2:]]
    # opening_contours = [BarPatternFeatures.contour_cosine(measure=x[0],key=x[1],n_beat=x[2]) for y in pieces for x in y[0:2]]
    # middle_contours = [BarPatternFeatures.contour_cosine(measure=x[0],key=x[1],n_beat=x[2]) for y in pieces for x in y[5:7]]

    # Experiment.generate_piece_from_tree(contour_coeff_dict={'opening': opening_contours, 'ending': ending_contours, 'middle':middle_contours })
    # Experiment.contour_self_similarity(D_major_pieces[0])
    # Experiment.recover_measures(specific_pieces[-1])
    # StreamBuilder.measures_to_stream(experiment.specific_pieces[-1])
    # Experiment.feature_evaluation_at_work(pieces=experiment.D_major_pieces[:20])
    # Experiment.cadence_detection_test(pieces=D_major_pieces[:5])
    # StreamBuilder.measures_to_stream(experiment.strong_candeces, title='strong candences')
    # StreamBuilder.measures_to_stream(experiment.weak_cadences, title='weak cadences')