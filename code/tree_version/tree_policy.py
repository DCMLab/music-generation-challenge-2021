from operation import Operation
from melody import Melody
import random
import numpy as np
from typing import Type


class Memory:
    def __init__(self,operation: Type[Operation]):
        self.operation = operation


class Action:
    def __init__(self, target_tree: Melody, operation: Type[Operation]):
        self.target_tree = target_tree
        self.operation = operation

    def is_legal(self):
        return self.operation.is_legal(self.target_tree)

    def perform(self):
        self.operation.perform(self.target_tree)
        self.target_tree.memory = Memory(operation=self.operation)

    def show(self):
        print(self.operation, 'on', self.target_tree.value)


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
        legal_actions = Policy._legal_actions(melody, operations)
        if not legal_actions:
            selected_action = None
        else:
            selected_action = random.choice(legal_actions)
        return selected_action


class BalancedTree(Policy):
    @staticmethod
    def determine_action(melody: Melody, operations: list[Type[Operation]]) -> Action:
        legal_actions = Policy._legal_actions(melody, operations)
        if not legal_actions:
            selected_action = None
        else:
            # expand the shallowest subtree in the starting subtrees
            current_subtree = melody
            continue_searching_condition = True
            while continue_searching_condition:
                children_sorted_by_depth = sorted(current_subtree.children, key=lambda tree: tree.get_depth())
                shallowest_children_with_legal_actions = \
                [subtree for subtree in children_sorted_by_depth if bool(set(subtree.get_surface()).intersection(
                    {action.target_tree for action in legal_actions}))][0]

                have_children = bool(shallowest_children_with_legal_actions.children)
                continue_dive_condition = have_children
                # print('current_subtree.value: ', current_subtree.value)
                # print('shallowest_children_with_legal_actions.value: ', shallowest_children_with_legal_actions.value)
                # print('have_children: ',have_children)
                # print('continue_dive_condition: ',continue_dive_condition)
                current_subtree = shallowest_children_with_legal_actions
                continue_searching_condition = have_children
                # print('continue_searching_condition: ', continue_searching_condition)
            # now we have found the subtree in the surface to expand
            print('subtree_to_expand.value: ', current_subtree.value)
            assert current_subtree in melody.get_surface()
            actions_for_current_subtree = [action for action in legal_actions if action.target_tree is current_subtree]
            assert bool(actions_for_current_subtree)
            selected_action = random.choice(actions_for_current_subtree)
        return selected_action
