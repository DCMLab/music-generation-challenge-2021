import copy
import itertools
import random

import numpy as np


def left_and_right_pitch(melody, slot):
    left_pitches = [x for i, x in enumerate(melody) if i < slot and x != '_']
    right_pitches = [x for i, x in enumerate(melody) if i > slot and x != '_']
    if len(left_pitches) > 0:
        left_pitch = left_pitches[-1]
    else:
        left_pitch = None
    if len(right_pitches) > 0:
        right_pitch = right_pitches[0]
    else:
        right_pitch = None
    # print('left_pitch, right_pitch: ', left_pitch, right_pitch)
    return left_pitch, right_pitch


class Operation:
    def __init__(self, type_of_operation):
        self.type_of_operation = None

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        pass

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        pass


class Neighbor(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        """works when left pitch = right pitch, both present"""
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        condition = all([
            left_pitch == right_pitch,
            left_pitch is not None,
            right_pitch is not None,
        ])
        return condition

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        scale = latent_info['scale']
        register = left_pitch//12
        neighbor_pitch = register +sorted(scale,key=lambda pitch: abs(pitch-left_pitch%12))[0]
        melody[slot:slot + 1] = '_', neighbor_pitch,'_'
        return melody


class Fill(Operation):
    def __init__(self):
        super().__init__(type_of_operation='Neighbor')

    @staticmethod
    def is_legal(melody: list, slot: int, latent_info: dict = None):
        """works when left pitch != right pitch, both present"""
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        condition = all([
            left_pitch is not None,
            right_pitch is not None,
            left_pitch != right_pitch,
        ])
        return condition

    @staticmethod
    def perform(melody: list, slot: int, latent_info: dict = None):
        left_pitch, right_pitch = left_and_right_pitch(melody, slot)
        midpoint = (right_pitch + left_pitch)/2
        low_pitch,high_pitch = sorted([left_pitch,right_pitch])
        scale_notes_between = [pitch for pitch in range(low_pitch,high_pitch+1) if pitch%12 in latent_info['scale']]
        print('scale_notes_between: ',scale_notes_between)
        fill_pitch = sorted(scale_notes_between,key=lambda pitch: [1/(1e-5+abs(pitch-midpoint))*(pitch%12 in latent_info['harmony'])])[-1]
        melody[slot:slot + 1] = '_', fill_pitch,'_'
        return melody



def elaborate_melody(melody: list, operations: list,latent_info: dict = None) -> list:
    print('----------------')
    slots = [i for i, x in enumerate(melody) if x == '_']
    legal_operations_on_slot = [(slot,operation) for slot,operation in itertools.product(slots,operations) if operation.is_legal(melody=melody, slot=slot)]
    print('legal_operations_on_slot: ',legal_operations_on_slot)
    selected_slot,selected_operation = random.choice(legal_operations_on_slot)
    print('selected_slot,selected_operation: ', (selected_slot,selected_operation))
    melody = selected_operation.perform(melody=melody,slot=selected_slot,latent_info=latent_info)
    print('resulted melody: ', melody)
    return melody

if __name__ == '__main__':
    operations = Operation.__subclasses__()
    print('set of operations: ', operations)

    print('\n**********\n')
    melody1 = [-5, '_', 7, '_',5]
    print('starting melody: ', melody1)
    latent_info1 = {'harmony': [0,4,7],
                   'scale': [0,2,4,5,7,9,11]}
    for i in range(3):
        elaborate_melody(melody1, operations,latent_info=latent_info1)

    print('\n**********\n')
    melody2 = [-5, '_', 5, '_', 4]
    print('starting melody: ', melody2)
    latent_info2 = {'harmony': [2, 7,11],
                   'scale': [0, 2, 4, 5, 7, 9, 11]}
    for i in range(3):
        elaborate_melody(melody2, operations, latent_info=latent_info2)
