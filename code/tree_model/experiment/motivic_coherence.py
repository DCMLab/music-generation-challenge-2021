from ..melody import Melody

import random
random.seed(1)

# local modules
import tree_policy
import ..operation
from template import pad_melody_templates
from form import Mperiod, Msentence, mperiod, msentence
from melody_tree_expansion import MelodyElaboration, PieceElaboration
import music21 as m21

elaborator = MelodyElaboration(operations=operation.Operation.__subclasses__(), policy=tree_policy.RhythmBalancedTree,
                               mimicking_policy=tree_policy.ImitatingPolicy)