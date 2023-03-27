# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:03:50 2022
"""

import csv
from collections import namedtuple


def write_curve_data(curve, filename):
    """ Writes the angles and Rs data into a csv file.
    """
    with open(filename+'.csv', 'w', newline='') as file:
        csvwriter = csv.writer(file)

        # field names
        fields = ['IntAngle', 'I Norm']
        csvwriter.writerow(fields)
        # data
        csvwriter.writerows([[ang, i_n]
                             for ang, i_n in zip(curve[0], curve[1])])


def read_curve_data(file_name):
    """ Reads a csv file for angles and Rs data and returns a
    Curve named tuple.
    """
    with open(file_name, newline='') as file:
        csvreader = csv.DictReader(file)

        x, y = [], []
        for row in csvreader:
            x += [float(row['IntAngle'])]
            y += [float(row['I Norm'])]

        Curve = namedtuple('Curve', 'x y')
        return Curve(x, y)


def read_curve_metricon(file_name, x_lim=(float('-inf'), float('inf'))):
    """ Reads a csv file for angles and Rs data and returns a
    Curve named tuple.
    """
    with open(file_name, newline='') as file:
        csvreader = csv.DictReader(file)

        x, y = [], []
        for row in csvreader:
            if (float(row['IntAngle']) > x_lim[0]) and (float(row['IntAngle']) < x_lim[1]):
                x += [float(row['IntAngle'])]
                y += [float(row['%I'])/100.]

        Curve = namedtuple('Curve', 'x y')
        return Curve(x, y)


def cutoff_data(curve, threshold):
    # Cutting of the data at higher values of y.
    curve_x = []
    curve_y = []
    for x, y in zip(curve.x, curve.y):
        if abs(1-y) > threshold:
            curve_x.append(x)
            curve_y.append(y)

    Curve = namedtuple('Curve', 'x y')
    return Curve(curve_x, curve_y)

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def read_config(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    fixed = {}
    bounds_dict = {}
    limits = (float('-inf'), float('inf'))

    for line in lines:
        line = line.strip() # remove any whitespace
        if line.startswith('fixed'):
            fixed_vars = line.split('=')[1].strip().split(',')
            
            for a in fixed_vars:
                var, param = a.split('(')
                param = param.strip(')')
                fixed[var.strip()] = float(param)

        elif line.startswith('bounds'):
            pairs_str = line.split('=')[1].strip()
            pairs = pairs_str.split(',')
            for a, b in pairwise(pairs):
                a = a.strip()
                b = b.strip()
                
                var, bound_1 = a.split('(')
                bound_2 = b.strip(')')
                bounds = (float(bound_1), float(bound_2))
                bounds_dict[var] = bounds
        elif line.startswith('limits'):
            limits = tuple(map(int, line.split('=')[1].strip().split(',')))
    
    return fixed, bounds_dict, limits
