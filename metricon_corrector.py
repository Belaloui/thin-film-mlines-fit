# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:24:55 2022
"""
from os import listdir
from os.path import isfile, join

def nth_repl(s, sub, repl, n):
    """ Code shamelessly stolen from the internet :)
    """
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s


filenames = [f for f in listdir('Metricon') if isfile(join('Metricon', f))]

print(filenames)

for filename in filenames:
    in_file = open('Metricon/'+filename, 'r')
    out_file = open('Metricon/Corrected/'+filename, 'w')

    lines = in_file.readlines()

    out_file.writelines(lines[2])

    for ind, line in enumerate(lines[3:]):
        if line.count(',') != 11:
            print(f'ERROR: line {ind} in file "{filename}".')
            print('"', line, '"')
            continue
        new_line = line
        for pos in [2, 3, 4, 6, 7]:
            new_line = nth_repl(new_line, ',', '.', pos)
        out_file.write(new_line)

    in_file.close()
    out_file.close()
