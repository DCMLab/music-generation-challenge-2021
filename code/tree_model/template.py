import copy
import random

from typing import List

from form import get_melody_templates_and_similarity_template
from melody import Melody


def pad_melody_templates(melody_templates: List[Melody], similarity_template: List[str]) -> Melody:
    # melody_templates = copy.deepcopy(_melody_templates)
    last_bar_has_tail = False
    self_similarity_template = similarity_template
    memory_padding = {}
    _melody_templates = melody_templates
    assert len(_melody_templates) == len(self_similarity_template)
    print(len(_melody_templates), len(self_similarity_template))
    for i, (melody_template, symbol) in enumerate(zip(_melody_templates, self_similarity_template)):
        if symbol in memory_padding.keys() and i != len(_melody_templates) - 1:
            add_what = memory_padding[symbol]
        else:
            if i != len(_melody_templates) - 1:
                if melody_template.no_tail:
                    add_what = 'none'
                else:
                    add_what = 'tail'
            else:
                add_what = random.choice(['none'])
        memory_padding[symbol] = add_what
        print(memory_padding)
        print('add_what: ', add_what)
        # add corresponding subtrees to head or tail
        if add_what == 'head':
            previous_bar = _melody_templates[i - 1]
            previous_note = copy.deepcopy(previous_bar.children[-1].transition[1])
            previous_note = previous_bar.children[-1].transition[1]
            first_note = melody_template.children[0].transition[0]
            new_transition = (previous_note, first_note)
            added_head = Melody(transition=new_transition, part='head')
            added_head.parent = melody_template
            melody_template.children = [added_head] + melody_template.children
        elif add_what == 'tail':
            # print('before padding:', len(melody_template.children))
            # print('i: ',i)

            next_bar = _melody_templates[i + 1]
            #next_note = copy.deepcopy(next_bar.children[0].transition[0])
            #last_note = copy.deepcopy(melody_template.children[-1].transition[1])
            next_note = next_bar.children[0].transition[0]
            last_note = melody_template.children[-1].transition[1]
            # print('pitch_cat of transition:', last_note.pitch_cat, next_note.pitch_cat)
            new_transition = (last_note, next_note)
            melody_template.add_children([Melody(transition=new_transition, part='tail')])
            # print('after padding:', len(melody_template.children))
        elif add_what == 'head_and_tail':
            next_bar = _melody_templates[i + 1]
            #next_note = copy.deepcopy(next_bar.children[0].transition[0])
            #last_note = copy.deepcopy(melody_template.children[-1].transition[1])
            next_note = next_bar.children[0].transition[0]
            last_note = melody_template.children[-1].transition[1]
            tail_transition = (last_note, next_note)
            previous_bar = melody_templates[i - 1]
            previous_note = previous_bar.children[-1].transition[1]
            first_note = melody_template.children[0].transition[0]
            head_transition = (previous_note, first_note)
            added_head = Melody(transition=head_transition, part='head')
            added_head.parent = melody_template
            melody_template.add_children([Melody(transition=tail_transition, part='tail')])
        elif add_what == 'none':
            pass
        else:
            assert False, add_what

    return _melody_templates

melody_templates, similarity_template = get_melody_templates_and_similarity_template()
padded_melody_templates = pad_melody_templates(melody_templates,similarity_template)


# for x in melody_templates:
#    print('&&&&&&&&&& unpadded')
#    x.show()
# for x in padded_melody_templates:
#    print('************ padded')
#    x.show()

# padded_old_templates = add_head_or_tail(old_templates=piece_old_templates)


# tree_templates = [old_template_to_tree(x) for x in padded_old_templates]

# print(tree_templates)
if __name__ == '__main__':
    pass
