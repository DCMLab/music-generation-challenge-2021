import itertools
import policy
import actions
from typing import Type


def elaborate_melody_one_step(melody: list, operations: list, latent_info: dict = None,
                              policy: Type[policy.Policy] = policy.UniformRandom) -> (list,dict):
    slots = [i for i, x in enumerate(melody) if x == '_']
    legal_operations_on_slot = [(slot, operation) for slot, operation in itertools.product(slots, operations) if
                                operation.is_legal(melody=melody, slot=slot,latent_info=latent_info)]
    if len(legal_operations_on_slot) == 0:
        raise Exception('legal_operations_on_slot is empty')
    selected_slot, selected_operation = policy.determine_action(state={'melody': melody},
                                                                legal_operations_on_slot=legal_operations_on_slot)
    new_melody = selected_operation.perform(melody=melody, slot=selected_slot, latent_info=latent_info)
    log = {'selected_slot': selected_slot,
           'selected_operation': selected_operation,
           'resulted_melody': new_melody,
           }
    return new_melody, log


melody_templates = [
    {
        'melody': [-5, '_', 7, '_', 5],
        'latent_info': {
            'harmony': [0, 4, 7],
            'scale': [0, 2, 4, 5, 7, 9, 11]
        }
    },

    {
        'melody': [-5, '_', 5, '_', 4],
        'latent_info': {
            'harmony': [2, 7, 11],
            'scale': [0, 2, 4, 5, 7, 9, 11]
        }
    },

    {
        'melody': [-5, '_', 7, '_', 5],
        'latent_info': {
            'harmony': [0, 4, 7],
            'scale': [0, 2, 4, 5, 7, 9, 11]
        }
    }
]


class Experiment:
    @staticmethod
    def elaborate_one_melody(melody_template: dict, n_iterations=3):
        operations = actions.Operation.__subclasses__()
        print('\n**********\n')
        print('set of operations: ', operations)
        melody, latent_info = melody_template.values()
        print('starting melody: ', melody)
        history = []
        for i in range(n_iterations):
            melody, log = elaborate_melody_one_step(melody, operations, latent_info=latent_info,
                                                        policy=policy.UniformRandom)
            history.append(log)

        for log in history:
            print(log)


if __name__ == '__main__':

    Experiment.elaborate_one_melody(melody_templates[0], n_iterations=3)
    Experiment.elaborate_one_melody(melody_templates[1], n_iterations=3)
