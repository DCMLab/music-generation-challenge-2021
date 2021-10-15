import itertools
import pprint
import music21 as m21

import policy
import actions
from visualize_helper import Converter
from typing import Type


def elaborate_melody_one_step(melody: list, operations: list, latent_info: dict = None,
                              policy: Type[policy.Policy] = policy.UniformRandom) -> (list, dict):
    slots = [i for i, x in enumerate(melody) if x == '_']
    legal_operations_on_slot = [(slot, operation) for slot, operation in itertools.product(slots, operations) if
                                operation.is_legal(melody=melody, slot=slot, latent_info=latent_info)]
    if len(legal_operations_on_slot) == 0:
        raise Exception('legal_operations_on_slot is empty')
    # first try the slot and operation from latent_info
    memory_nonempty = {'memory_slot', 'memory_operation'}.issubset(latent_info.keys())
    memory_compatible = None
    memory_slot = None
    memory_operation = None
    if memory_nonempty:
        memory_slot = latent_info['memory_slot']
        memory_operation = latent_info['memory_operation']
        memory_compatible = (memory_slot, memory_operation) in legal_operations_on_slot
    if memory_compatible is True:
        print(':D reusing memory_slot and memory_operation')
        selected_slot, selected_operation = memory_slot, memory_operation
    else:
        if memory_compatible is False:
            print(':( memory_slot and memory_operation is not legal')
        selected_slot, selected_operation = policy.determine_action(state={'melody': melody},
                                                                    legal_operations_on_slot=legal_operations_on_slot)
    new_melody = selected_operation.perform(melody=melody, slot=selected_slot, latent_info=latent_info)
    log = {'legal_operations_on_slot': legal_operations_on_slot,
           'selected_slot': selected_slot,
           'selected_operation': selected_operation,
           'resulted_melody': new_melody,
           }
    return new_melody, log


beginning = {
    'melody': [12, '_', 12, '_', 12],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

second = {
    'melody': [11, '_', 9, '_', 7, '_', 5, '_', 4, '_', 2],
    'latent_info': {
        'harmony': [2, 7, 11],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

seq_1 = {
    'melody': [-5, '_', 7, '_', 5],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

seq_2 = {
    'melody': [-5, '_', 5, '_', 4],
    'latent_info': {
        'harmony': [2, 7, 11],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

pre_cadence = {
    'melody': [0, '_', 7, '_', 7],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

cadence = {
    'melody': [4, '_', 2, '_', 0],
    'latent_info': {
        'harmony': [0, 4, 7],
        'scale': [0, 2, 4, 5, 7, 9, 11]
    }
}

melody_templates = [beginning,
                    second,
                    seq_1,
                    seq_2,
                    seq_1,
                    seq_2,
                    pre_cadence,
                    cadence]


class Experiment:
    @staticmethod
    def elaborate_one_melody(melody_template: dict, n_iterations=3, memory: list[dict] = None, policy=None):
        operations = actions.Operation.__subclasses__()
        print('\n**********\n')
        print('set of operations: ', operations)
        melody, latent_info = melody_template['melody'], melody_template['latent_info']
        print('starting melody: ', melody)
        history = []
        for i in range(n_iterations):
            if memory is not None:
                memory_slot = memory[i]['selected_slot']
                memory_operation = memory[i]['selected_operation']
                latent_info.update({'memory_slot': memory_slot})
                latent_info.update({'memory_operation': memory_operation})
            melody, log = elaborate_melody_one_step(melody, operations, latent_info=latent_info,
                                                    policy=policy)
            history.append(log)
        for log in history:
            print('------')
            pprint.pprint(log,width=100,sort_dicts=False)

        return history

    @staticmethod
    def generate_first_8_bars(list_of_melody_templates, similarity_template, policy=policy.UniformRandom):
        created_memories = {}
        new_melodies = []
        for melody_template, sim_symbol in zip(list_of_melody_templates, similarity_template):
            print('created_memories.keys(): ', created_memories.keys())
            if sim_symbol not in created_memories.keys():
                memory = Experiment.elaborate_one_melody(melody_template, n_iterations=6, policy=policy)
                created_memories.update({sim_symbol: memory})
            else:

                memory = Experiment.elaborate_one_melody(melody_template, n_iterations=6,
                                                         memory=created_memories[sim_symbol], policy=policy)
            new_melody = memory[-1]['resulted_melody']
            new_melodies.append(new_melody)
        return new_melodies


if __name__ == '__main__':
    similarity_template = ['a', 'b', 'c', 'c', 'c', 'c', 'd', 'e']
    new_melodies = Experiment.generate_first_8_bars(list_of_melody_templates=melody_templates,
                                                    similarity_template=similarity_template,
                                                    policy=policy.UniformRandom)
    #print(new_melodies)
    measures = [Converter.melody_list_to_m21_measure(new_melody) for new_melody in new_melodies]
    stream = m21.stream.Stream()
    stream.append(measures)
    stream.show()
