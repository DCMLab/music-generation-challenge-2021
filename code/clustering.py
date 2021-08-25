import collections
import random
import sys

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

xml_files = ['../data/xml/' + file_name for file_name in
             filter(lambda x: x.endswith('.xml'), sorted(os.listdir('../data/xml')))]
pieces = []
samples = random.sample(xml_files, k=100)

specific_sample = ['../data/xml/Polonäs_6b9196.xml',
                   '../data/xml/_Polska_efter_Petter_Dufva_75c046.xml',
                   '../data/xml/_Polonäs_sexdregasamlingen_del_2_nr_40_f6ee90.xml',
                   '../data/xml/Polonäs_i_Dm_efter_Daniel_Danielsson_ac3754.xml']
for xml_file in samples:
    test_piece = m21.converter.parse(xml_file)
    voice = test_piece.getElementsByClass(m21.stream.Part)[0]
    measures = voice.getElementsByClass(m21.stream.Measure)
    key_signature = test_piece.flat.getKeySignatures()[0]
    time_signature = test_piece.flat.getTimeSignatures()[0]
    measures_with_piece_info = [(m, key_signature, time_signature) for m in measures]
    pieces.append(measures_with_piece_info)
    if len(measures_with_piece_info) == 8:
        print(xml_file)
all_measures = sum(pieces, [])

twelve_bar_pieces = [x for x in pieces if len(x) == 8]

assert len(twelve_bar_pieces) > 0


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
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.beat] for note in notes]
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
        notes = list(measure.getElementsByClass(m21.note.Note))
        scale_degrees = [note.pitch.pitchClass - key.tonic.pitchClass for note in notes]
        diff_scale_degrees = [0]
        diff_scale_degrees.extend([scale_degrees[i] - scale_degrees[i - 1] for i in range(1, len(scale_degrees))])
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.offset, note.duration.quarterLength,
                  diff_scale_degrees[i]] for i, note in
                 enumerate(notes)]
        grid = np.full(12, fill_value='_', dtype=str)
        for i, note in enumerate(notes):
            index = int(note[1] * 4)
            if index >= 12:
                print('encountered a non 3/4 bar')
                pass
            else:
                grid[index] = 'x'
                duration = int(note[2] * 4)
                grid[index + 1:index + duration + 1] = '-'
        return grid

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
        notes = []
        for event in events:
            if type(event) == m21.chord.Chord:
                note = event.notes[-1]
                notes.append(note)
            elif type(event) == m21.note.Note:
                note = event
                notes.append(note)
            else:
                print('encountered neither chord or note: ', type(event), 'disgard event')
                pass

        notes = [[note.pitch.midi - key.tonic.midi, note.offset, note.duration.quarterLength] for i, note in
                 enumerate(notes)]
        grid = np.full(n_grid, fill_value=999, dtype=object)
        for i, note in enumerate(notes):
            index = int(note[1] * 4)
            if index >= 12:
                print('encountered a non 3/4 bar, disgard')
                pass
            else:
                refined_index = int(index * n_grid / 12)
                grid[refined_index] = note[0]
                duration = int(note[2] * 4 * n_grid / 12)
                refined_duration = int(duration * n_grid / 12)
                grid[refined_index + 1:refined_index + refined_duration + 1] = note[0]
        for i, x in enumerate(grid):
            if x == 999:
                grid[i] = grid[i - 1]
        if np.count_nonzero(grid == 999) > 0:
            if np.alltrue(grid == 999):
                pass
            else:
                print('encountered rest in grid: ', grid)

        return grid

    @staticmethod
    def contour_cosine(measure: music21.stream.Measure, key: music21.key.KeySignature,
                       n_beat) -> np.ndarray:
        ## cosine
        contour_grid = BarPatternFeatures.contour_sixteenth_grid(measure, key, n_beat, n_grid=120)
        dct = scipy.fftpack.dct(contour_grid, norm='forward')
        dct = dct[:25]
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


class Plot:
    @staticmethod
    def draw_scatter(distance_matrix, text_arrays, file_name, model):
        plotly.io.templates.default = "plotly_white"

        fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                            specs=[[{"type": "scene"}],
                                                   ])
        for i, perplexity in enumerate([40]):
            tsne = sklearn.manifold.TSNE(n_components=3, metric="precomputed", perplexity=perplexity, learning_rate=20,
                                         square_distances=True)
            embeded = tsne.fit_transform(distance_matrix)
            # text_list = np.squeeze(text_arrays)
            text_list = [' '.join([str(y) for y in x]) for x in text_arrays]
            # text_list = [str(np.array(x)).replace('\n', '<br>') for x in text_list]
            scatter = go.Scatter3d(x=embeded[:, 0], y=embeded[:, 1], z=embeded[:, 2], mode='markers',
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
    def draw_heatmap(distance_matrix):
        plotly.io.templates.default = "plotly_white"

        fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                            specs=[[{"type": "scene"}],
                                                   ])
        display_array = np.squeeze(measure_feature).tolist()
        hovertext = list()
        for yi, yy in enumerate(display_array):
            hovertext.append(list())
            for xi, xx in enumerate(display_array):
                hovertext[-1].append('bar 1: {}<br />bar 2: {}<br />distance: {}'.format(xx, yy, Levenshtein[yi][xi]))
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
    def draw_dct(dcts):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 2)
        ax.plot(np.average(np.abs(dcts), axis=0))
        plt.show()


Mode = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 9, 10, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
}


class MelodySynthesis:
    @staticmethod
    def naive_combine_contour_rhythm_vl(contour_coeffs, rhythm_grid, vl, sample_point_size, mode=Mode['major']):
        """without using harmonic content"""
        approx_contour = scipy.fftpack.idct(contour_coeffs, norm='backward', n=sample_point_size)
        print('approx_contour: ', approx_contour)
        pitch_grid = []
        for i, x in enumerate(rhythm_grid):
            if x == 'x':
                scale = np.array(mode)
                tiled_scale = np.array([scale + 12 * i for i in np.arange(-1, 3)]).reshape(-1)
                xs = np.linspace(0, 11, num=sample_point_size)
                approx_contour_16_grid = approx_contour[np.argmin(np.abs(xs - (i + 0.25)))]
                diff = np.abs(tiled_scale - approx_contour_16_grid)
                argmin = np.argmin(diff)
                pitch = tiled_scale[argmin]
            elif x == '-':
                pitch = pitch_grid[i - 1]
            else:
                pitch = 999
            pitch_grid.append(pitch)
        return pitch_grid

    @staticmethod
    def combine_contour_rhythm_vl(contour_coeffs, rhythm_grid, vl, sample_point_size, pc_distribution_grid):
        """use harmonic content ended in pc_distribution_grid, which has shape=(12,12), meaning (#16th-notes,#pc)"""
        approx_contour = scipy.fftpack.idct(contour_coeffs, norm='backward', n=sample_point_size)
        print('approx_contour: ', approx_contour)
        pc_distribution_grid = pc_distribution_grid + 0.1
        pitch_grid = []
        for i, x in enumerate(rhythm_grid):
            if x == 'x':
                print('pc_distribution_grid.shape:', pc_distribution_grid.shape)
                tiled_pc_distribution_grid = np.tile(pc_distribution_grid[i], 3)  # shape = (3*12,)
                print('tiled_pc_distribution_grid: ', tiled_pc_distribution_grid)
                print('tiled_pc_distribution_grid.shape:', tiled_pc_distribution_grid.shape)
                xs = np.linspace(0, 11, num=sample_point_size)
                approx_contour_16_grid = np.average(
                    approx_contour[np.argmin(np.abs(xs - (i + 0))):np.argmin(np.abs(xs - (i + 0.25)))])  # shape = (,)
                print('approx_contour_16_grid.shape:', approx_contour_16_grid.shape)
                print('approx_contour_16_grid:', approx_contour_16_grid)
                tiled_pitches = np.arange(tiled_pc_distribution_grid.shape[0]) - 12  # shape = (3*12,)
                print('tiled_pitches:', tiled_pitches)
                print('tiled_pitches.shape:', tiled_pitches.shape)
                diff = np.abs(tiled_pitches - approx_contour_16_grid)  # shape = (3*12,1)
                print('diff.shape:', diff.shape)
                force = np.power(tiled_pc_distribution_grid, 2) / (diff + 1e-5)  # shape = (3*12,1)
                print('diff:', diff)
                print('np.power(tiled_pc_distribution_grid[i],2): ', np.power(tiled_pc_distribution_grid[i], 2))
                print('force:', force)
                print('force.shape:', force.shape)
                argmax = np.argmax(force)  # shape = (,)
                print('argmax.shape:', argmax.shape)
                pitch = tiled_pitches[argmax]
            elif x == '-':
                pitch = pitch_grid[i - 1]
            else:
                pitch = 999
            pitch_grid.append(pitch)
        return pitch_grid


def pitch_grid_rhythm_grid_to_stream(pitch_grid: list, rhythm_grid: list,
                                     key: m21.key.KeySignature) -> m21.stream.Measure:
    stream = m21.stream.Measure()
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

    for note in note_list:
        stream.append(note)
    return stream


class Experiment:
    mode_pc_onehot = {
        'major': np.tile([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], (12, 1)),
        'minor': np.tile([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], (12, 1)),
    }

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

        pc_distribution_grid = Experiment.mode_pc_onehot[Mm]
        for test_measure in measures:
            stream.append(test_measure[0])
            test_contour = BarPatternFeatures.contour_cosine(test_measure[0], key=key,
                                                             n_beat=test_measure[2].numerator)
            test_rhythm = BarPatternFeatures.rhythm_sixteenth_grid(test_measure[0], key=key,
                                                                   n_beat=test_measure[2].numerator)
            constructed_melody = MelodySynthesis.combine_contour_rhythm_vl(contour_coeffs=test_contour,
                                                                           rhythm_grid=test_rhythm, vl=None,
                                                                           sample_point_size=300,
                                                                           pc_distribution_grid=pc_distribution_grid)
            constructed_stream = pitch_grid_rhythm_grid_to_stream(constructed_melody, test_rhythm, key=key)
            print(constructed_melody)
            print(test_rhythm)
            constructed_melodies.append(constructed_melody)
            stream_constructed_melodies.append(constructed_stream)

        print(constructed_melodies)
        stream.show()
        stream_constructed_melodies.show()
        sys.exit()

    @staticmethod
    def mix_match_rhythm_contour(first_piece_measures: list[m21.stream.Measure],
                                 second_piece_measures: list[m21.stream.Measure]):

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

            constructed_stream1 = pitch_grid_rhythm_grid_to_stream(constructed_melody1, rhythm2, key=key1)

            constructed_stream2 = pitch_grid_rhythm_grid_to_stream(constructed_melody2, rhythm1, key=key2)

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
    def texture_contour_diff(all_measures: list[m21.stream.Measure]):

        all_pitches = []
        all_contour = []
        all_coeff = []
        print('len(first_piece_measures): ', len(all_measures))
        for measure in all_measures:
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
            all_contour.append([contour_approxs_downbeat, contour_diff_4_8, contour_diff_16_8])
            all_coeff.append([contour_downbeat, contour_coe_diff_4_8, contour_coe_diff_16_8])
            all_pitches.append(pitches)
        # draw_contour(all_pitches,all_contour,n_sample_points=300)
        return all_pitches, all_contour, all_coeff

    @staticmethod
    def clustering_contour_diff(all_measures):
        all_pitches = [BarPatternFeatures.contour_sixteenth_grid(measure[0], key=measure[1],
                                                                 n_beat=measure[2].numerator) for measure in
                       all_measures]
        contours_incremental_metrical = np.array(Experiment.texture_contour_diff(all_measures)[2])
        print(contours_incremental_metrical.shape)
        print('calculating AgglomerativeClustering')
        model = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=3, affinity='precomputed',
                                                linkage='average')
        print('calculating distance')

        diff = np.expand_dims(contours_incremental_metrical, axis=1) - np.expand_dims(contours_incremental_metrical,
                                                                                      axis=0)
        _D = np.linalg.norm(diff, axis=-1, ord=2)
        print('_D.shape: ', _D.shape)
        metrical_weight = np.array([1, 1, 1])
        print('_D: ', _D)
        weighted__D = _D * metrical_weight
        print('weighted__D: ', weighted__D)
        D = np.linalg.norm(weighted__D, axis=-1, ord=2)

        print('D.shape: ', D.shape)
        print('unique distance percentage:',
              2 * np.unique(D).shape[0] / (D.size - D.shape[0]))
        model.fit(D)
        text_array = all_pitches

        draw_scatter(distance_matrix=D, text_arrays=text_array, file_name='contour metrical', model=model)
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
        draw_scatter(distance_matrix=Levenshtein, text_arrays=text_array, file_name='contour')


if __name__ == '__main__':
    # Experiment.recover_measures(measures=twelve_bar_pieces[1])
    # Experiment.mix_match_rhythm_contour(twelve_bar_pieces[2], twelve_bar_pieces[3])
    Experiment.clustering_contour_diff(all_measures)
