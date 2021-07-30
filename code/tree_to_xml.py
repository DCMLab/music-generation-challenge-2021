import music21 as m21
from piece import Tree

note_list = []
def get_note_level(tree: Tree) -> list:
    current_node = tree
    if current_node.children == []:
        infos = (current_node.data.harmony.symbols[0], current_node.data.rhythm.symbols[0],current_node.data.contour.symbols[0],current_node.data.form.symbols[0])
        note_list.append(infos)
    else:
        for i,child in enumerate(current_node.children):
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
    for harmony,rhythm,contour,form in note_list:
        str_rhythm = str(rhythm)
        if str_rhythm in note_rhythm.keys():
            note_like = m21.note.Note('A4')
            note_like.quarterLength = note_rhythm[str_rhythm]
        elif str_rhythm in rest_rhythm.keys():
            note_like = m21.note.Rest()
            note_like.quarterLength =rest_rhythm[str_rhythm]
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
            harmony,rhythm,contour,form=(current_node.data.harmony.symbols[0], current_node.data.rhythm.symbols[0],current_node.data.contour.symbols[0],current_node.data.form.symbols[0])
            str_rhythm = str(rhythm)
            if str_rhythm in note_rhythm.keys():
                note_like = m21.note.Note('A4')
                note_like.quarterLength = note_rhythm[str_rhythm]
            elif str_rhythm in rest_rhythm.keys():
                note_like = m21.note.Rest()
                note_like.quarterLength = rest_rhythm[str_rhythm]
            assert note_like is not None

            #note_like.addLyric(harmony)
            #note_like.addLyric(contour)
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
    annotated_stream = add_annotation(unannotated_stream,tree)
    stream.append(annotated_stream)
    return stream

def add_annotation(stream:m21.stream.Stream,tree:Tree)->m21.stream.Stream:
    current_node = tree
    current_stream = stream
    _harmony, _rhythm, _contour, _form = (
        current_node.data.harmony.symbols[0], current_node.data.rhythm.symbols[0], current_node.data.contour.symbols[0],
        current_node.data.form.symbols[0])
    note_to_annotate = current_stream.flat[0]
    if current_node.data.level in ['Bar','Note','Beat']:
        if current_node.data.level == 'Bar':
            contour = str(_contour)
            harmony = str(_harmony)
            form = str(_form)
            #note_to_annotate.addLyric(contour)
            #note_to_annotate.addLyric(harmony)
            note_to_annotate.insertLyric(form,2)
        if current_node.data.level == 'Beat':
            contour = str(_contour)
            harmony = str(_harmony)
            #note_to_annotate.addLyric(contour)
            note_to_annotate.insertLyric(harmony,1)

        if current_node.data.level == 'Note':
            contour = str(_contour)
            note_to_annotate.insertLyric(contour,0)


    if current_node.data.level == 'Note':
        pass
    else:
        for tree_child,stream_child in zip(tree.children,stream):
            add_annotation(stream_child,tree_child)
    return stream



if __name__ == '__main__':
    from piece import Tree, node_grow

    test_tree = Tree()

    node_grow(tree=test_tree, level='Top')
    node_grow(tree=test_tree, level='Section')
    node_grow(tree=test_tree, level='Phrase')
    node_grow(tree=test_tree, level='Bar')
    node_grow(tree=test_tree, level='Beat')
    print(test_tree)

    stream = tree_to_stream_powerful(test_tree)
    stream.show()
    stream.show('text')


