"""
TODO:

test tournament -> club ranking method
"""
import numpy as np

from rankings import (
    Match,
    Club,
    League,
    GoalBasedLeague,
    RankingMethod,
    ScoresRanking,
    MaximumLikelihood,
    Neustadtl,
    FairBets,
    Buchholz,
    RecursivePerformance,
    RecursiveBuchholz,
    GeneralisedRowSum,
)
from outputs import rescale

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

def test_league():
    a = "team a"
    b = "team b"
    c = "the team of c"
    matches = [
        Match(home=a, away=b, result=[1, 1]),
        Match(home=a, away=c, result=[1, 4]),
        Match(home=b, away=c, result=[4, 4]),
        Match(home=c, away=b, result=[2, 3]),
    ]
    league = League(matches, abbreviations={a: "TA", b: "TB", c: "TC"})

    # check clubs
    assert list(c.name for c in league.clubs) == [a, b, c]
    a_club = league.clubs[0]
    b_club = league.clubs[1]
    c_club = league.clubs[2]
    assert a_club.abbrev == "TA"
    assert a_club.played == 2
    assert a_club.won == 0
    assert a_club.drawn == 1
    assert a_club.lost == 1
    assert a_club.goals_for == 2
    assert a_club.goals_against == 5
    assert a_club.goal_difference == -3
    assert a_club.points == 1
    assert a_club.home_games == 2

    assert b_club.abbrev == "TB"
    assert b_club.played == 3
    assert b_club.won == 1
    assert b_club.drawn == 2
    assert b_club.lost == 0
    assert b_club.goals_for == 8
    assert b_club.goals_against == 7
    assert b_club.goal_difference == 1
    assert b_club.points == 5
    assert b_club.home_games == 1

    assert c_club.abbrev == "TC"
    assert c_club.played == 3
    assert c_club.won == 1
    assert c_club.drawn == 1
    assert c_club.lost == 1
    assert c_club.goals_for == 10
    assert c_club.goals_against == 8
    assert c_club.goal_difference == 2
    assert c_club.points == 4
    assert c_club.home_games == 1

    # check results matrix
    assert np.all(league.results_matrix == np.array([
        [0, 1, 0],
        [1, 0, 4],
        [3, 1, 0]
    ]))

    # check results matrix for goal-based tournament
    goal_league = GoalBasedLeague(matches)
    assert np.all(goal_league.results_matrix == np.array([
        [0, 1, 1],
        [1, 0, 7],
        [4, 6, 0]
    ]))

def test_ordinal_ranking():
    a = Club(name="a FC", club_id=0)
    b = Club(name="b FC", club_id=1)
    c = Club(name="c FC", club_id=2)
    d = Club(name="d FC", club_id=3)
    scores = [(a, 0.4), (b, -7), (c, 6), (d, 0.41)]
    assert RankingMethod.ordinal_ranking(scores) == [c, d, a, b]

def test_rescale():
    tests = (
        (np.array([1,2,3,4]), (10, 11), np.array([10, 10 + 1/3, 10 + 2/3, 11])),
        (np.array([-1, 0, 3]), (1, 9), np.array([1, 3, 9])),
    )
    for xs, (mn, mx), exp in tests:
        assert np.all(rescale(xs, mn, mx) == exp), f"failure for {xs}"
