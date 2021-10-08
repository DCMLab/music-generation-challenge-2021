import copy

import music21 as m21
import numpy as np

from piece import Tree

note_list = []


def get_note_level(tree: Tree) -> list:
    current_node = tree
    if current_node.children == []:
        infos = (
        current_node.data.harmony.symbols[0], current_node.data.rhythm.symbols[0], current_node.data.contour.symbols[0],
        current_node.data.form.symbols[0])
        note_list.append(infos)
    else:
        for i, child in enumerate(current_node.children):
            current_node = child
            get_note_level(current_node)
    return note_list


note_rhythm = {
    'h': 2.0,
    'q': 1.0,
    'e': 0.5,
    's': 0.25,
    'edot': 0.75,
}

rest_rhythm = {
    'r_q': 1.0
}


def tree_to_stream(tree: Tree) -> m21.stream.Stream:
    stream = m21.stream.Stream()
    stream.append(m21.key.Key('D'))
    stream.append(m21.meter.TimeSignature('3/4'))
    stream.append(m21.metadata.Metadata(title='A Note-Level Tree', composer='Tree'))
    note_list = get_note_level(tree=tree)
    for harmony, rhythm, contour, form in note_list:
        str_rhythm = str(rhythm)
        if str_rhythm in note_rhythm.keys():
            note_like = m21.note.Note('A4')
            note_like.quarterLength = note_rhythm[str_rhythm]
        elif str_rhythm in rest_rhythm.keys():
            note_like = m21.note.Rest()
            note_like.quarterLength = rest_rhythm[str_rhythm]
        note_like.addLyric(harmony)
        note_like.addLyric(contour)
        note_like.addLyric(form)
        stream.append(note_like)
    return stream


def tree_to_stream_powerful(tree: Tree) -> m21.stream.Stream:
    def create_nested_stream_from_tree(tree: Tree) -> m21.stream.Stream:
        current_node = tree
        sub_stream = m21.stream.Stream()
        if current_node.data.level == 'Note':
            harmony, rhythm, contour, form = (current_node.data.harmony.symbols[0], current_node.data.rhythm.symbols[0],
                                              current_node.data.contour.symbols[0], current_node.data.form.symbols[0])
            str_rhythm = str(rhythm)
            if str_rhythm in note_rhythm.keys():
                note_like = m21.note.Note('A4')
                note_like.quarterLength = note_rhythm[str_rhythm]
            elif str_rhythm in rest_rhythm.keys():
                note_like = m21.note.Rest()
                note_like.quarterLength = rest_rhythm[str_rhythm]
            assert note_like is not None

            # note_like.addLyric(harmony)
            # note_like.addLyric(contour)
            sub_stream.append(note_like)
        else:
            for child in tree.children:
                sub_stream.append(create_nested_stream_from_tree(child))
        return sub_stream

    stream = m21.stream.Stream()
    stream.append(m21.key.Key('D'))
    stream.append(m21.meter.TimeSignature('3/4'))
    stream.append(m21.metadata.Metadata(title='Note-level Tree with Analysis', composer='Tree'))
    unannotated_stream = create_nested_stream_from_tree(tree)
    annotated_stream = add_annotation(unannotated_stream, tree)
    stream.append(annotated_stream)
    return stream


def add_annotation(stream: m21.stream.Stream, tree: Tree) -> m21.stream.Stream:
    current_node = tree
    current_stream = stream
    _harmony, _rhythm, _contour, _form = (
        current_node.data.harmony.symbols[0], current_node.data.rhythm.symbols[0], current_node.data.contour.symbols[0],
        current_node.data.form.symbols[0])
    note_to_annotate = current_stream.flat[0]
    if current_node.data.level in ['Subphrase', 'Bar', 'Note', 'Beat']:
        if current_node.data.level == 'Subphrase':
            contour = str(_contour)
            harmony = str(_harmony)
            form = str(_form)
            # note_to_annotate.addLyric(contour)
            # note_to_annotate.addLyric(harmony)
            note_to_annotate.insertLyric(form, 2)
        if current_node.data.level == 'Beat':
            contour = str(_contour)
            harmony = str(_harmony)
            # note_to_annotate.addLyric(contour)
            note_to_annotate.insertLyric(harmony, 1)

        if current_node.data.level == 'Note':
            contour = str(_contour)
            note_to_annotate.insertLyric(contour, 0)

    if current_node.data.level == 'Note':
        pass
    else:
        for tree_child, stream_child in zip(tree.children, stream):
            add_annotation(stream_child, tree_child)
    return stream


class PieceInfo:
    default_shape = (4, 8, 2, 2)
    default_scale = [0, 2, 4, 5, 7, 9, 11]
    default_time_signature = '3/4'

    def __init__(self, shape=default_shape, scale=default_scale, time_signature=default_time_signature, template=None,
                 target_chords=None):
        self.shape = shape
        self.scale = scale
        self.template = template
        self.time_signature = time_signature
        self.target_chords = None


def tree_to_piece_info(tree: Tree) -> PieceInfo:
    first_section = tree.children[0]
    first_phrase = first_section.children[0]
    bars = sum([subphrase.children for subphrase in first_phrase.children], [])
    shape = (3, len(bars), 3, 2)
    template = np.empty(shape=shape, dtype='object')
    template.fill('*')
    target_chords = np.transpose(copy.deepcopy(template), (1, 2, 3, 0))
    template[:, :-1, :, 0] = '~'
    template[:, -1, 0, 0] = '~'
    template[0, [0, -1], 0, 0] = '12'
    template[-1, [0, -1], 0, 0] = '0'

    chord_string_to_array_dict = {
        '|I': [0, 4, 7],
        'I': [0, 4, 7],
        'I(AC)': [0, 4, 7],
        'ii': [2, 5, 9],
        'V': [2, 7, 11]
    }

    for _bar, bar in enumerate(bars):
        for _beat, beat in enumerate(bar.children):
            chord_string = str(beat.data.harmony.symbols[0])
            chord_array = chord_string_to_array_dict[chord_string]
            target_chords[_bar, _beat, :, :] = chord_array
    # print(template)
    piece_info = PieceInfo()
    piece_info.template = template.tolist()
    piece_info.target_chords = target_chords[np.newaxis,...]
    piece_info.shape = shape
    piece_info.time_signature = '3/4'
    return piece_info

from piece import Tree, node_grow
if __name__ == '__main__':
    from piece import Tree, node_grow

    test_tree = Tree()

    node_grow(tree=test_tree, level='Top')
    # print(test_tree)
    node_grow(tree=test_tree, level='Section')
    # print(test_tree)
    node_grow(tree=test_tree, level='Phrase')
    # print(test_tree)
    node_grow(tree=test_tree, level='Subphrase')
    # print(test_tree)
    node_grow(tree=test_tree, level='Bar')
    # print(test_tree)
    node_grow(tree=test_tree, level='Beat')
    print(test_tree)

    tree_to_piece_info(test_tree)

    stream = tree_to_stream_powerful(test_tree)
    stream.show()

    # stream.write('xml',fp='./')

    # stream.show('text')
