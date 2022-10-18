"""
Contains a named tuple used to represent the path statistic results from a Monte Carlo simulation.
"""
from collections import namedtuple

PathStatistics = \
    namedtuple('PathStatistics',
               ['TheoreticalMean', 'EmpiricalMean', 'TheoreticalStandardDeviation', 'EmpiricalStandardDeviation'])
