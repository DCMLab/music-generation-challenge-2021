import copy
import random

import music21.stream
from typing import Type
# local modules
import tree_policy
from melody import Melody
import operation
import string
from template import pad_melody_templates
from form import Mperiod,Msentence,mperiod,msentence
from melody_tree_expansion import MelodyElaboration,PieceElaboration
import time
elaborator = MelodyElaboration(operations=operation.Operation.__subclasses__(), policy=tree_policy.RhythmBalancedTree,
                               mimicking_policy=tree_policy.ImitatingPolicy)

from datetime import datetime



for myform in [Mperiod,Msentence,mperiod,msentence]:
    for i in range(2):
        piece_elaborator = PieceElaboration(elaborator, tree_templates=pad_melody_templates(myform.to_melody_templates(),myform.to_similarity_template()),
                                            self_similarity_template=myform.to_similarity_template())
        piece_elaborator.elaborate()
        stream = piece_elaborator.result_to_stream()

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y %H:%M:%S"+random.choice(string.ascii_uppercase))
        stream.write(fp=dt_string,fmt='musicxml')

if __name__ == '__main__':
    pass