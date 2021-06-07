"""
TODO:

test tournament -> club ranking method
"""
from io import StringIO

import numpy as np

from rankings import *
from outputs import rescale
from providers import CSVProvider
from utils import kendall_tau_distance

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
    mbd = [
        [
            Match(home=a, away=b, result=[1, 1])
        ],
        [
            Match(home=a, away=c, result=[1, 4]),
            Match(home=b, away=c, result=[4, 4])
        ],
        [
            Match(home=c, away=b, result=[2, 3])
        ],
    ]
    fixtures = Fixtures(matches_by_date=mbd)
    assert fixtures.all_matches() == mbd[0] + mbd[1] + mbd[2]
    assert fixtures.num_dates == 3
    # test fixture subsetting
    half = fixtures.partial(0.5)
    assert len(half.matches_by_date) == 1
    assert half.matches_by_date == fixtures.matches_by_date[0:1]
    half = fixtures.partial(2 / 3)
    assert len(half.matches_by_date) == 2
    assert half.matches_by_date == fixtures.matches_by_date[0:2]

    # test league
    league = League(fixtures, abbreviations={a: "TA", b: "TB", c: "TC"})

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
    goal_league = GoalBasedLeague(fixtures)
    assert np.all(goal_league.results_matrix == np.array([
        [0, 1, 1],
        [1, 0, 7],
        [4, 6, 0]
    ]))

def test_league_club_names():
    fixtures = Fixtures(matches_by_date=[[
        Match(home="a", away="b", result=[1, 1])
    ]])
    l1 = League(fixtures)
    l2 = League(fixtures, club_names=["a", "b", "c"])

    assert len(l1.clubs) == 2
    assert len(l2.clubs) == 3

    a1, b1 = l1.clubs
    a2, b2, c2 = l2.clubs
    assert a1.name == "a"
    assert a2.name == "a"
    assert b1.name == "b"
    assert b2.name == "b"
    assert c2.name == "c"
    assert a2.played == 1
    assert b2.played == 1
    assert c2.played == 0

def test_ordinal_ranking():
    a = Club(name="a FC", club_id=0)
    b = Club(name="b FC", club_id=1)
    c = Club(name="c FC", club_id=2)
    d = Club(name="d FC", club_id=3)

    class MyRankingMethod(RankingMethod):
        def rank(self, _):
            return np.array([1,4,1,3])

    class MyTieBreaker(RankingMethod):
        def rank(self, _):
            return np.array([10,11,12,13])

    league = League(Fixtures(matches_by_date=[]), club_names=["a","b","c","d"])
    expected = ["b", "d", "c", "a"]
    ranking = MyRankingMethod().ordinal_ranking(
        league, tie_breakers=[MyTieBreaker()]
    )
    assert [c.name for c in ranking] == expected

def test_rescale():
    tests = (
        (np.array([1,2,3,4]), (10, 11), np.array([10, 10 + 1/3, 10 + 2/3, 11])),
        (np.array([-1, 0, 3]), (1, 9), np.array([1, 3, 9])),
    )
    for xs, (mn, mx), exp in tests:
        assert np.all(rescale(xs, mn, mx) == exp), f"failure for {xs}"

def test_csv_conversion():
    provider = CSVProvider(
        "the-date-of-the-game",
        ("%d/%m/%y", "%d/%m/%Y"),
        "team-that-played-at-home",
        "team-that-played-away",
        "goals-for-the-home-team",
        "goals-for-the-away-team",
    )
    csv_lines = [
        "Blah,the-date-of-the-game,team-that-played-at-home,"
        "team-that-played-away,goals-for-the-home-team,"
        "goals-for-the-away-team",
        "_,05/05/2020,Joe FC,Bob United,4,0",
        "_,05/05/2020,Bill FC,Dave Albion,1,2",
        "_,12/05/2020,Bill FC,Joe FC,3,4",
        "_,13/05/2020,Dave Albion,Bob United,4,5",
        "_,13/05/2020,AFC Steve,Joe FC,0,9",
    ]
    buf = StringIO()
    buf.write("\n".join(csv_lines))
    buf.seek(0)
    fixtures = provider.csv_to_fixtures(buf)
    assert len(fixtures.matches_by_date) == 3
    w1, w2, w3 = fixtures.matches_by_date
    assert w1 == [
        Match(home="Joe FC", away="Bob United", result=(4, 0)),
        Match(home="Bill FC", away="Dave Albion", result=(1, 2))
    ]
    assert w2 == [Match(home="Bill FC", away="Joe FC", result=(3, 4))]
    assert w3 == [
        Match(home="Dave Albion", away="Bob United", result=(4, 5)),
        Match(home="AFC Steve", away="Joe FC", result=(0, 9))
    ]

def test_reducible():
    # reducible: cannot get from index 0 to 2
    A_red = np.array([
        [1, 0, 2],
        [5, 0, 1],
        [1, 0, 0],
    ])
    # irreducible
    A_irred = np.array([
        [1, 3, 2],
        [5, 0, 1],
        [1, 0, 0],
    ])
    assert TournamentRanking.is_reducible(A_red)
    assert not TournamentRanking.is_reducible(A_irred)

def test_kendall_tau_distance():
    l1 = ["a", "b", "c", "d", "e"]
    l2 = ["c", "d", "a", "b", "e"]
    # swaps are {a, c}, {a, d}, {b, c}, {b, d}, so distance should be 4
    assert kendall_tau_distance(l1, l2) == 4
