# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:28:42 2022
"""

import csv
import matplotlib.pyplot as plt

from collections import namedtuple


def read_curves_data(file_name):
    """ Reads a csv file for angles and Rs data and returns a
    Curve named tuple.
    """
    with open(file_name, newline='') as file:
        csvreader = csv.DictReader(file)

        ext_ang, int_ang, intens, intens_norm, intens_norm_smth =\
            [], [], [], [], []

        for row in csvreader:
            ext_ang += [float(row['ExtAngle'])]
            int_ang += [float(row['IntAngle'])]
            intens += [float(row['I'])]
            intens_norm += [float(row['%I'])/100.]
            intens_norm_smth += [float(row['Smooth %I'])/100.]

        Curves = namedtuple('Curve', 'ext_ang int_ang intens '
                            'intens_norm intens_norm_smth')
        return Curves(ext_ang, int_ang, intens, intens_norm, intens_norm_smth)


curves = read_curves_data('Data/T3 TE.txt')
# curves = read_curves_data('Metricon/Corrected/T1 TE.txt')

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(curves.int_ang, curves.intens_norm_smth)
ax1.set_xlabel("IntAngle")

ax1.grid()
plt.show()
