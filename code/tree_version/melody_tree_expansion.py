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
from pc_helpers import interval_list_to_pitch_list


class MelodyElaboration:
    def __init__(self, operations: list[Type[operation.Operation]], policy: Type[tree_policy.Policy],
                 elaboration_to_mimic=None):
        self.operations = operations
        self.policy = policy

    def elaborate_one_step(self, melody: Melody, show=True):
        selected_action = self.policy.determine_action(melody, self.operations)
        if selected_action is not None:
            selected_action.perform()
            if show is True:
                selected_action.show()
        else:
            if show is True:
                print('no action is available, do not perform elaboration')

    def elaborate(self, melody: Melody, steps=3, show=True):
        for i in range(steps):
            print('---', 'step', i + 1, '---')
            self.elaborate_one_step(melody, show)
            if show is True:
                melody.show()





class PieceElaboration:
    def __init__(self, melody_elaborator: MelodyElaboration, tree_templates: list[Melody],
                 self_similarity_template: list[str] = None):
        self.melody_elaborator = melody_elaborator
        self.tree_templates = tree_templates
        self.trees = copy.deepcopy(tree_templates)
        self.self_similarity_template = self_similarity_template

    def elaborate(self):
        for i, melody in enumerate(self.trees):
            print('******', 'bar', i + 1, '******')
            self.melody_elaborator.elaborate(melody, steps=4, show=True)

    def result_to_stream(self):
        stream = music21.stream.Stream()
        surfaces = [melody.get_surface() for melody in self.trees]
        surfaces_values = [[x.value for x in surface] for surface in surfaces]
        pitch_lists = [interval_list_to_pitch_list(surfaces_value) for surfaces_value in surfaces_values]
        measures = [Converter.melody_list_to_m21_measure(pitch_list) for pitch_list in pitch_lists]
        # stream.append([Converter.melody_list_to_m21_measure(interval_list_to_pitch_list(melody.get_surface())) for melody in piece_elaborator.trees])
        stream.append(measures)
        return stream


if __name__ == '__main__':
    elaborator = MelodyElaboration(operations=operation.Operation.__subclasses__(), policy=tree_policy.BalancedTree)
    piece_elaborator = PieceElaboration(elaborator, tree_templates=template.tree_templates)
    piece_elaborator.elaborate()
    stream = piece_elaborator.result_to_stream()
    stream.show()
