"""
A named tuple used to represent pricing results from a Monte Carlo simulation.
"""
from collections import namedtuple


MonteCarloPricingResults = namedtuple('MonteCarloPricingResults', ['price', 'error'])
