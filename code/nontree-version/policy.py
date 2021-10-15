import collections

from actions import Operation
import random


class Policy:
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        pass


class UniformRandom(Policy):
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        selected_slot, selected_operation = random.choice(legal_operations_on_slot)
        return selected_slot, selected_operation

class PreferNeighbor(Policy):
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        print('legal_operations_on_slot: ',legal_operations_on_slot)
        legal_neighbor_on_slot = [x for x in legal_operations_on_slot if x[-1].__name__ == 'Neighbor']
        if len(legal_neighbor_on_slot)>0:
            selected_slot, selected_operation = random.choice(legal_neighbor_on_slot)
        else:
            selected_slot, selected_operation = random.choice(legal_operations_on_slot)
        return selected_slot, selected_operation

class PreferFill(Policy):
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        print('legal_operations_on_slot: ',legal_operations_on_slot)
        legal_fill_on_slot = [x for x in legal_operations_on_slot if x[-1].__name__ == 'Fill']
        if len(legal_fill_on_slot)>0:
            coin = random.uniform(0,1)
            print(coin)
            if coin>1:
                selected_slot, selected_operation = random.choice(legal_fill_on_slot)
            else:
                selected_slot, selected_operation = random.choice(legal_operations_on_slot)
        else:
            selected_slot, selected_operation = random.choice(legal_operations_on_slot)
        return selected_slot, selected_operation

class EachPositionEqualChance(Policy):
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        #print('legal_operations_on_slot: ', legal_operations_on_slot)
        position_counts = collections.Counter([x[0]for x in legal_operations_on_slot])
        N = len(legal_operations_on_slot)
        weights = [1/(N*position_counts[x[0]]) for x in legal_operations_on_slot]
        #print(weights)
        selected_slot, selected_operation = random.choices(legal_operations_on_slot,weights=weights,k=1)[0]
        return selected_slot, selected_operation

class EachOperationEqualChance(Policy):
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        operation_counts = collections.Counter([x[-1]for x in legal_operations_on_slot])
        N = len(legal_operations_on_slot)
        weights = [1/(N*operation_counts[x[-1]]) for x in legal_operations_on_slot]
        selected_slot, selected_operation = random.choices(legal_operations_on_slot,weights=weights,k=1)[0]
        return selected_slot, selected_operation

class EvenExpansion(Policy):
    """uses the expansion history to balance slot elaboration"""
    @staticmethod
    def determine_action(state: dict, legal_operations_on_slot) -> (int, Operation):
        raise NotImplementedError