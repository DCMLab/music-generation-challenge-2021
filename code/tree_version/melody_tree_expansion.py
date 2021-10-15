import copy

import music21.stream

import tree_policy
from melody import Melody
import Operation
import template
from typing import Type
from visualize_helper import Converter

class MelodyElaboration:
    def __init__(self, operations: list[Type[Operation.Operation]], policy: Type[tree_policy.Policy]):
        self.operations = operations
        self.policy = policy

    def elaborate_one_step(self,melody:Melody,show=True):
        selected_action = self.policy.determine_action(melody,self.operations)
        if selected_action is not None:
            selected_action.perform()
            if show is True:
                selected_action.show()
        else:
            if show is True:
                print('no action is available, do not perform elaboration')

    def elaborate(self,melody:Melody,steps=3,show=True):
        for i in range(steps):
            print('---','step',i+1,'---')
            self.elaborate_one_step(melody,show)
            if show is True:
                melody.show()

def interval_list_to_pitch_list(interval_list:list[(int,int)]) -> list[int]:
    pitch_list = []
    for i,pair in enumerate(interval_list):
        if i == 0:
            pitch_list.extend(pair)
        else:
            pitch_list.append(pair[1])
    return pitch_list

class PieceElaboration:
    def __init__(self,melody_elaborator:MelodyElaboration,tree_templates: list[Melody]):
        self.melody_elaborator = melody_elaborator
        self.tree_templates = tree_templates
        self.trees = copy.deepcopy(tree_templates)
    def elaborate(self):
        for i,melody in enumerate(self.trees):
            print('******','bar',i+1,'******')
            self.melody_elaborator.elaborate(melody,steps=4,show=True)

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
    elaborator = MelodyElaboration(operations=Operation.Operation.__subclasses__(), policy=tree_policy.UniformRandom)
    piece_elaborator = PieceElaboration(elaborator,tree_templates=template.tree_templates)
    piece_elaborator.elaborate()
    stream = piece_elaborator.result_to_stream()
    stream.show()
