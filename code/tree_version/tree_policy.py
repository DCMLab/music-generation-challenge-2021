import copy

from operation import Operation
from melody import Melody
import random
import numpy as np
from typing import Type


class Memory:
    def __init__(self, operation: Type[Operation]):
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
        print(self.operation, 'on', self.target_tree.transition,self.target_tree.transition[0].latent_variables)


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
                # print('current_subtree.value: ', current_subtree.value)
                # print('shallowest_children_with_legal_actions.value: ', shallowest_children_with_legal_actions.value)
                # print('have_children: ',have_children)
                # print('continue_dive_condition: ',continue_dive_condition)
                current_subtree = shallowest_children_with_legal_actions
                continue_searching_condition = have_children
                # print('continue_searching_condition: ', continue_searching_condition)
            # now we have found the subtree in the surface to expand
            print('subtree_to_expand.transition: ', current_subtree.transition)
            assert current_subtree in melody.get_surface()
            actions_for_current_subtree = [action for action in legal_actions if action.target_tree is current_subtree]
            print('operations_for_current_subtree: ',[x.operation for x in actions_for_current_subtree])
            assert bool(actions_for_current_subtree)
            selected_action = random.choice(actions_for_current_subtree)
        return selected_action

class RhythmBalancedTree(Policy):
    @staticmethod
    def determine_action(melody: Melody, operations: list[Type[Operation]]) -> Action:
        legal_actions = Policy._legal_actions(melody, operations)
        if not legal_actions:
            selected_action = None
        else:
            durations = [min(x.target_tree.transition[0].rhythm_cat,x.target_tree.transition[1].rhythm_cat) for x in legal_actions]
            selected_action = random.choices(legal_actions,weights=np.array(durations)**2)[0]
            print('durations: ',durations)
            print('np.argmax(durations): ',np.argmax(durations))
            #selected_action = legal_actions[np.argmax(durations)]
            print(selected_action.__dict__)
        return selected_action


class ImitatingPolicy:

    @staticmethod
    def _matching_subtree_pairs(melody: Melody, memory_melody: Melody) -> list[(Melody, Melody)]:
        matching_subtree_pairs = []
        found_a_match = (not melody.children)
        if found_a_match:
            matching_subtree_pairs.append((melody,memory_melody))
        else:
            same_num_children = len(melody.children) == len(memory_melody.children)
            print('length of melody/memory_melody .children: ',len(melody.children),len(memory_melody.children))
            print('same_num_children: ',same_num_children)
            if same_num_children:
                for child_melody, child_memory_melody in zip(melody.children,memory_melody.children):
                    new_pairs = ImitatingPolicy._matching_subtree_pairs(child_melody,child_memory_melody)
                    matching_subtree_pairs.extend(new_pairs)
        return matching_subtree_pairs

    @staticmethod
    def determine_action(melody: Melody, operations: list[Type[Operation]], memory_melody: Melody) -> Action:
        """imitating the memory melody as far as possible"""
        memory_melody = copy.deepcopy(memory_melody)
        matching_subtree_pairs = ImitatingPolicy._matching_subtree_pairs(melody, memory_melody)
        print('there are matching_subtree_pairs: ', bool(matching_subtree_pairs))
        matching_subtree_pairs_with_legal_operations = [pair for pair in matching_subtree_pairs if
                                                        [operation.is_legal(pair[0]) for operation in operations]]
        matching_subtree_pairs_with_legal_operations_memory_has_children = [pair for pair in matching_subtree_pairs_with_legal_operations if bool(pair[1].children)]
        dists_to_root = [x[0].get_dist_to_root() for x in matching_subtree_pairs_with_legal_operations_memory_has_children]
        #print('dist_to_root: ',dists_to_root)
        matching_subtree_pairs_with_legal_operations_memory_has_children = sorted(matching_subtree_pairs_with_legal_operations_memory_has_children,key=lambda x:x[0].get_dist_to_root())
        if bool(matching_subtree_pairs_with_legal_operations_memory_has_children):
            current_melody_tree, current_memory_melody_tree = matching_subtree_pairs_with_legal_operations_memory_has_children[0]
            if current_memory_melody_tree.memory.operation.is_legal(current_melody_tree):
                selected_operation = current_memory_melody_tree.memory.operation
            else:
                print('unable to imitate memory operation',current_memory_melody_tree.memory.operation)
                legal_operations = [operation for operation in operations if operation.is_legal(current_melody_tree)]
                selected_operation = legal_operations[0]
                print('instead, using legal_operation', selected_operation)
            selected_action = Action(current_melody_tree, selected_operation)
        else:
            print('unable to imitate: there is no matching subtrees ')
            selected_action = None
        return selected_action
