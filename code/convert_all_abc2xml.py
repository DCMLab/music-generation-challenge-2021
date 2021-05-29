#!/usr/bin/env python
#coding=utf-8

import argparse, os
from abc2xml_231.abc2xml import getXmlScores


def abc2xml(abc):
    return abc

def read_abc(path):
    with open(path, 'r', encoding='utf-8') as f:
        abc = f.read()
    return abc

def main(args):
    for subdir, _, files in os.walk(args.DIR):
        for file in files:
            print(file.encode('utf-8','surrogatepass').decode('utf-8'))
            fname, fext = os.path.splitext(file)
            if fext != '.abc':
                continue
            src = os.path.join(subdir, file)
            tgt = os.path.join(subdir, fname + '.xml')
            try:
                print(f"{src} => {tgt}")
            except:
                print(subdir)
                print(file)

# f = '../data/abc/_1814_698087.abc'
# read_abc(f)
#
# from abc2xml_231.abc2xml import readfile, getXmlScores
# t = readfile(f)
# xml = getXmlScores(read_abc(f))
#
# def write_xml(xml, path):
#     with open(path, 'w', encoding='utf-8') as f:
#         f.write(xml)
#
# write_xml(xml[0], '../data/xml/_1814_698087.xml')


def check_dir(d):
    """ Turn input into an existing, absolute directory path.
    """
    if not os.path.isdir(d):
        d = os.path.join(os.getcwd(),d)
        if not os.path.isdir(d):
            if input(d + ' does not exist. Create? (y|n)') == "y":
                os.makedirs(d)
            else:
                raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    if not os.path.isabs(d):
        d = os.path.abspath(d)
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = '''\
---------------------------------
| Convert a folder of abc files |
---------------------------------
''')
    parser.add_argument('DIR', type=check_dir, help='Folder containing abc files.')
    parser.add_argument('TARGET_DIR', nargs='?', type=check_dir, default=os.getcwd(), help='Output folder for converted XML files. Defaults to current working directory.')
    args = parser.parse_args()
    main(args)
