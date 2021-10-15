from Operation import Operation
from melody import Melody
import random
from typing import Type


class Action:
    def __init__(self, target_tree: Melody, operation: Type[Operation]):
        self.target_tree = target_tree
        self.operation = operation

    def is_legal(self):
        return self.operation.is_legal(self.target_tree)

    def perform(self):
        self.operation.perform(self.target_tree)

    def show(self):
        print(self.operation,'on',self.target_tree.value)


class Policy:
    """ Given a melody tree and a list of candidate actions determine where and what operation to use"""

    @staticmethod
    def _legal_actions(melody: Melody, operations: list[Type[Operation]]) -> list[Action]:
        """ return a list of all legal actions"""
        surface_subtrees = melody.get_surface()
        all_actions = [Action(subtree, operation) for subtree in surface_subtrees for operation in operations]
        legal_actions = [action for action in all_actions if action.is_legal()]
        return legal_actions

    @staticmethod
    def determine_action(melody: Melody, operations: list[Type[Operation]]) -> Action:
        """ return a operation and the subtree from the surface on which the operation will be done"""
        pass


class UniformRandom(Policy):
    @staticmethod
    def determine_action(melody: Melody, operations: list[Type[Operation]]) -> Action:
        legal_actions = Policy._legal_actions(melody,operations)
        if not legal_actions:
            selected_action = None
        else:
            selected_action = random.choice(legal_actions)
        return selected_action
