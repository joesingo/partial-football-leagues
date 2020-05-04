"""
TODO:

test league table
test average points
test tournament -> club ranking method
"""
import numpy as np

from rankings import (
    ScoresRanking,
    MaximumLikelihood,
    Neustadtl,
    FairBets,
    Buchholz,
    RecursivePerformance,
    RecursiveBuchholz,
    GeneralisedRowSum,
)

def test_ranking_methods():
    # from example 2.1 of González-Díaz
    A = np.array([[0, 4, 5], [1, 0, 25], [0, 1, 0]])
    expected = (
        (ScoresRanking, np.array([0.9, 0.839, 0.032])),
        (MaximumLikelihood, np.array([2.051, 0.608, -2.659])),
        (Neustadtl, np.array([0.352, 0.055, 0.027])),
        (FairBets, np.array([0.801, 0.192, 0.006])),
        (Buchholz, np.array([1.335, 1.011, 0.881])),
        (RecursivePerformance, np.array([1.764, 0.491, -2.255])),
        (RecursiveBuchholz, np.array([0.267, 0.086, -0.353])),
        (GeneralisedRowSum, np.array([0.39, 0.407, -0.798])),
    )
    for cls, exp in expected:
        got = cls()._rank(A)
        assert np.all(np.round(got, 3) == exp), f"failure for {cls.__name__}"
