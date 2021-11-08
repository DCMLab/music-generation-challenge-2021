# external libs
import copy
import music21.stream
from typing import Type
# local modules
import tree_policy
from melody import Melody
import operation
import template
from visualize_helper import Converter
from pc_helpers import interval_list_to_pitch_list,melody_surface_to_pitch_list


class MelodyElaboration:
    def __init__(self, operations: list[Type[operation.Operation]], policy: Type[tree_policy.Policy], mimicking_policy: Type[tree_policy.ImitatingPolicy],
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
        for i in range(steps):
            print('---', 'step', i + 1, '---')
            self.elaborate_one_step(melody, memory_melody,show)
            if show is True:
                melody.show()


class PieceElaboration:
    def __init__(self, melody_elaborator: MelodyElaboration, tree_templates: list[Melody],
                 self_similarity_template: list[str] = None):
        self.melody_elaborator = melody_elaborator
        self.tree_templates = tree_templates
        self.trees = copy.deepcopy(tree_templates)
        self.self_similarity_template = self_similarity_template
        self.symbol_memory = {}

    def elaborate(self):
        steps = 4
        for i, melody in enumerate(self.trees):
            print('\n******', 'bar', i + 1, '******\n ')
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

    def result_to_stream(self):
        stream = music21.stream.Stream()
        stream.append(music21.meter.TimeSignature('3/4'))
        measures = [tree.surface_to_stream() for tree in self.trees]
        stream.append(measures)
        return stream


if __name__ == '__main__':
    elaborator = MelodyElaboration(operations=operation.Operation.__subclasses__(), policy=tree_policy.RhythmBalancedTree,mimicking_policy=tree_policy.ImitatingPolicy)
    piece_elaborator = PieceElaboration(elaborator, tree_templates=template.padded_melody_templates,
                                        self_similarity_template=template.handcoded_similarity)
    piece_elaborator.elaborate()
    stream = piece_elaborator.result_to_stream()
    stream.show()
