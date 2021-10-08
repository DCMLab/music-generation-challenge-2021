
import os

import corpus_analyzer

abc_files = ['../data/abc/' + file_name for file_name in sorted(os.listdir('../../data/abc'))]
xml_files = ['../data/xml/' + file_name for file_name in filter(lambda x:x.endswith('generated_template_for_polyphony.xml'), sorted(os.listdir(
    '../../data/xml')))]



def n_sections(abc_string):
    # list_string = abc_string.splitlines()
    # print(list_string[10:])
    new_abc_string = str(abc_string).strip()
    end_repeats = new_abc_string.count(':|') + new_abc_string.count('::')
    if end_repeats == 0:
        n = 1
    elif new_abc_string[-1:] != ':|':
        n = end_repeats + 1
    else:
        n = end_repeats

    return n

def abc_scratch():
    abc_strings = [open(file).read() for file in abc_files[:]]
    number_sections = [n_sections(abc_string) for abc_string in abc_strings]
    print(collections.Counter(number_sections))
    for i, x in enumerate(abc_strings):
        if n_sections(x) == 1:
            pass










if __name__ == '__main__':
    corpus_analyzer.Analyzer.beat_level_similarity(xml_files[6:7])