# external libs

import copy
import music21.stream
from typing import Type, List,Tuple
# local modules
import tree_policy
from melody import Melody
import operation
import template
import form

from pc_helpers import interval_list_to_pitch_list,melody_surface_to_pitch_list




class MelodyElaboration:
    def __init__(self, operations: List[Type[operation.Operation]], policy: Type[tree_policy.Policy], mimicking_policy: Type[tree_policy.ImitatingPolicy],
                 melody_template: Melody = None, rhythm_template: Melody = None):
        self.operations = operations
        self.policy = policy
        self.mimicking_policy = mimicking_policy
        print('memory: ', melody_template)

    def elaborate_one_step(self, melody: Melody,memory_melody:Melody, show=True):
        # whether mimicking memory_melody to enforce coherence
        if memory_melody is None:
            selected_action = self.policy.determine_action(melody, self.operations)
        else:
            selected_action = self.mimicking_policy.determine_action(melody,self.operations,memory_melody)

        if selected_action is not None:
            selected_action.perform()
            selected_action.show()
        else:
            print('no action is available, do not perform elaboration')

    def elaborate(self, melody: Melody, memory_melody:Melody=None, steps=3, show=True):
        melody.history = [copy.deepcopy(melody)]
        for i in range(steps):
            print('---', 'step', i + 1, '---')
            self.elaborate_one_step(melody, memory_melody,show)
            melody.history.append(copy.deepcopy(melody))
            if show is True:
                melody.show()


class PieceElaboration:
    def __init__(self, melody_elaborator: MelodyElaboration, tree_templates: List[Melody],
                 self_similarity_template: List[str] = None):
        self.melody_elaborator = melody_elaborator
        self.tree_templates = tree_templates
        self.trees = copy.deepcopy(tree_templates)
        self.self_similarity_template = self_similarity_template
        self.symbol_memory = {}

    def elaborate(self):
        for i, melody in enumerate(self.trees):
            print('\n******', 'bar', i + 1, '******\n ')
            n =10
            steps = min(n,melody.max_elaboration)
            if self.self_similarity_template is not None:
                current_symbol = self.self_similarity_template[i]
                if current_symbol in self.symbol_memory.keys():
                    print('loading memory \'{}\'\n'.format(current_symbol))
                    memory_melody = self.symbol_memory[current_symbol]
                    #self.melody_elaborator.memory_melody = memory_melody
                    self.melody_elaborator.elaborate(melody, steps=steps, show=True,memory_melody=memory_melody)
                    #self.melody_elaborator.memory_melody = None
                else:
                    print('writing memory \'{}\'\n'.format(current_symbol))
                    self.melody_elaborator.elaborate(melody, steps=steps, show=True)
                    self.symbol_memory.update({current_symbol: melody})
            else:
                self.melody_elaborator.elaborate(melody, steps=steps, show=True)

    def surface_to_stream(self):
        stream = music21.stream.Stream()
        stream.append(music21.tempo.MetronomeMark(number=100, referent=1.))
        stream.append(music21.meter.TimeSignature('3/4'))
        measures = [tree.surface_to_stream() for tree in self.trees]
        stream.append(measures)
        return stream

    def history_to_stream(self):
        streams = music21.stream.Stream()
        streams.append(music21.metadata.Metadata(title='The elaboration process',composer='Interval tree model'))


        longest_history_length = max([len(tree.history) for tree in self.trees])
        for i in range(longest_history_length):
            stream = music21.stream.Part()
            stream.append(music21.metadata.Metadata())
            stream.partName = f'step {i}'
            stream.append(music21.tempo.MetronomeMark(number=100, referent=1.))
            stream.append(music21.meter.TimeSignature('3/4'))
            if i>0:
                measures = [tree.history[min(i,len(tree.history)-1)].surface_to_stream(last_iteration_stream=streams.getElementsByClass(music21.stream.Part)[-1].getElementsByClass(music21.stream.Measure)[j]) for j,tree in enumerate(self.trees)]
            else:
                measures = [tree.history[min(i, len(tree.history) - 1)].surface_to_stream() for j, tree in enumerate(self.trees)]
            stream.append(measures)
            streams.append(stream)
        return streams




if __name__ == '__main__':
    import random
    #random.seed(1)
    elaborator = MelodyElaboration(operations=operation.Operation.__subclasses__(), policy=tree_policy.RhythmBalancedTree,mimicking_policy=tree_policy.ImitatingPolicy)
    myform=form.mperiod
    piece_elaborator = PieceElaboration(elaborator,
                                        tree_templates=template.pad_melody_templates(myform.to_melody_templates(),
                                                                                     myform.to_similarity_template()),
                                        self_similarity_template=myform.to_similarity_template())
    piece_elaborator.elaborate()
    stream = piece_elaborator.history_to_stream()

    stream.show()
