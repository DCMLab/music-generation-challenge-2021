import random

random.seed(1)

# local modules
import tree_policy
import operation
from template import pad_melody_templates
from form import Mperiod, Msentence, mperiod, msentence
from melody_tree_expansion import MelodyElaboration, PieceElaboration
import music21 as m21

elaborator = MelodyElaboration(operations=operation.Operation.__subclasses__(), policy=tree_policy.RhythmBalancedTree,
                               mimicking_policy=tree_policy.ImitatingPolicy)

from datetime import datetime

template_names = ['Major period', 'Major sentence', 'minor period', 'minor sentence']
for i, myform in enumerate([Mperiod, Msentence, mperiod, msentence]):
    for j in range(250):
        piece_elaborator = PieceElaboration(elaborator,
                                            tree_templates=pad_melody_templates(myform.to_melody_templates(),
                                                                                myform.to_similarity_template()),
                                            self_similarity_template=myform.to_similarity_template())
        print('template_names[i]: ', template_names[i])
        piece_elaborator.elaborate()
        stream = piece_elaborator.surface_to_stream()
        if i in [0, 1]:
            interval = m21.interval.Interval('M2')
            print('interval: ', interval.show('text'))
            stream = stream.transpose(interval)
            stream.keySignature = m21.key.KeySignature(2)
            stream.makeAccidentals(inPlace=True)
        file_name = '{}-{}'.format(template_names[i], j)
        stream.metadata = m21.metadata.Metadata(title=file_name, composer='Interval tree model')

        xml_folder_path = 'generated_xml/'
        midi_folder_path = 'generated_midi/'
        stream.write(fp=xml_folder_path + file_name, fmt='musicxml')
        stream.write(fp=midi_folder_path + file_name + '.mid', fmt='midi')

if __name__ == '__main__':
    pass
