import random

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

xml_files = ['../data/xml/' + file_name for file_name in
             filter(lambda x: x.endswith('.xml'), sorted(os.listdir('../data/xml')))]
all_measures = []
for xml_file in xml_files[:100]:
    test_piece = m21.converter.parse(xml_file)
    measures = test_piece.getElementsByClass(m21.stream.Part)[0].getElementsByClass(m21.stream.Measure)
    all_measures.extend(measures)


class BarPatternFeatures:
    @staticmethod
    def scale_degree_and_onset(measure: music21.stream.Measure, key: music21.key.KeySignature, n_beat) -> np.ndarray:
        """ output shape = (n_beat,4,2) representing (beat_index,16th_index,[scale_degree,event:oneset or hold or offset]) """
        # array = np.zeros(shape=(n_beat,4,1,1))
        notes = list(measure.getElementsByClass(m21.note.Note))
        notes = [[note.pitch.pitchClass - key.tonic.pitchClass, note.duration.quarterLength] for note in notes]
        return notes


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


def levenshtein_with_operation_cost(_seq1, _seq2):
    def insert_cost(seq):
        duration = seq[1]
        return (4*duration)*(4*duration)

    def substitution_cost(seq1, seq2):
        pitch_diff = 12*abs(seq2[0] - seq1[0])
        duration_diff = 4*abs(seq2[1] - seq1[1])
        return pitch_diff*duration_diff

    def deletion_cost(seq):
        duration = seq[1]
        return (4*duration)*(4*duration)

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


def test():
    sample_points = np.random.random(100).reshape((-1, 2))
    sample_points = np.round(sample_points, 2) * 100
    print(sample_points)
    model = cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(sample_points)
    return model, sample_points


def generate_plot(model, y, labels=None):
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(y, truncate_mode=None, labels=labels, orientation='left')
    plt.xlabel("Distance")
    plt.show()


if __name__ == '__main__':
    key_signature = test_piece.flat.getKeySignatures()[0]
    time_signature = test_piece.flat.getTimeSignatures()[0]
    measure_arrays = [
        BarPatternFeatures.scale_degree_and_onset(measure, key=key_signature, n_beat=time_signature.numerator) for
        measure in all_measures]
    measure_arrays = np.array(measure_arrays, dtype=object).reshape(-1, 1)
    _Levenshtein = distance.pdist(measure_arrays, levenshtein_with_operation_cost)
    Dfunc = lambda x: distance.pdist(x, levenshtein_with_operation_cost)
    Levenshtein = distance.squareform(_Levenshtein)
    print(Levenshtein)

    model = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=3, affinity='precomputed',
                                            linkage='average')
    model.fit(Levenshtein)
    linkage_matrix = linkage(_Levenshtein)
    # generate_plot(model,y=linkage_matrix,labels=np.squeeze(measure_arrays))
    # print(model.labels_)
    # print(measure_arrays.shape)
    plotly.io.templates.default = "plotly_white"
    fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                        specs=[[{"type": "scene"}],
                                               ])

    def draw_heatmap():
        display_array = np.squeeze(measure_arrays).tolist()
        hovertext = list()
        for yi, yy in enumerate(display_array):
            hovertext.append(list())
            for xi, xx in enumerate(display_array):
                hovertext[-1].append('bar 1: {}<br />bar 2: {}<br />distance: {}'.format(xx, yy, Levenshtein[yi][xi]))

        fig.add_trace(
            go.Heatmap(z=Levenshtein, hovertemplate='%{text}<extra></extra>',text=hovertext,showscale=False),
            row=1, col=1
        )

    def draw_scatter():

        for i,perplexity in enumerate([50]):
            tsne = sklearn.manifold.TSNE(n_components=3, metric="precomputed", perplexity=perplexity, learning_rate=5,square_distances=True)
            embeded = tsne.fit_transform(Levenshtein)
            fig.add_trace(go.Scatter3d(x=embeded[:, 0], y=embeded[:, 1], z=embeded[:, 2], mode='markers',
                                       marker=dict(
                                           size=6,
                                           color=model.labels_,  # set color to an array/list of desired values
                                           colorscale='Phase',
                                           opacity=1
                                       ), hovertemplate='%{text}',
                                       text=np.squeeze(measure_arrays).tolist()),
                          row=i+1, col=1)

    draw_scatter()
    fig.write_html('heat and scatter.html')
    # fig = plotly.figure_factory.create_dendrogram(measure_arrays,distfun=Dfunc,hovertext=measure_arrays.tolist(),labels=None)
    # fig.write_html('denfrogram.html'.format(perplexity))
