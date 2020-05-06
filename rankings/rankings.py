from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import itertools
import math
import operator

import numpy as np
import numpy.linalg as linalg
from sympy import Matrix

from utils import listify

@dataclass(eq=True)
class Match:
    home: str
    away: str
    result: Tuple[int, int]

@dataclass
class Fixtures:
    matches_by_date: List[List[Match]]

    @listify
    def all_matches(self) -> List[Match]:
        for matches in self.matches_by_date:
            yield from matches

    def partial(self, t: float):
        """
        Return a new Fixtures object containing only matches up to part-way
        through the season, according to the parameter t in [0, 1]
        """
        n = math.floor(t * len(self.matches_by_date))
        return Fixtures(matches_by_date=self.matches_by_date[:n])

    @property
    def num_dates(self):
        return len(self.matches_by_date)

@dataclass
class Club:
    """
    Representation of a club in a league
    """
    name: str
    club_id: int
    abbrev: Optional[str] = None

    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_games: int = 0

    @property
    def played(self):
        return self.won + self.drawn + self.lost

    @property
    def goal_difference(self):
        return self.goals_for - self.goals_against

    @property
    def points(self):
        return 3 * self.won + self.drawn

    def register_match(self, scored, conceded, home=True):
        self.goals_for += scored
        self.goals_against += conceded
        if scored > conceded:
            self.won += 1
        elif scored < conceded:
            self.lost += 1
        else:
            self.drawn += 1

        if home:
            self.home_games += 1

class League:
    """
    Representation of a (partial) league. Consists of a collection of Clubs and
    the tournament results matrix
    """
    clubs: List[Club]
    results_matrix: np.ndarray

    def __init__(self, fixtures: Fixtures, abbreviations=None,
                 club_names=None):
        abbreviations = abbreviations or {}
        matches = fixtures.all_matches()

        self.clubs = []
        # build list of clubs
        name_to_club = {}
        names = club_names or self.get_club_names(matches)
        for name in names:
            club = Club(
                name=name,
                abbrev=abbreviations.get(name),
                club_id=None  # will be set after sorting
            )
            name_to_club[club.name] = club
            self.clubs.append(club)

        # sort alphabetically and set correct IDs
        self.clubs.sort(key=lambda c: c.abbrev or c.name)
        for i, club in enumerate(self.clubs):
            club.club_id = i

        # construct tournament matrix
        self.results_matrix = np.zeros((self.num_clubs, self.num_clubs))
        for match in matches:
            home = name_to_club[match.home]
            away = name_to_club[match.away]

            # record the match details in the Club objects
            home_goals, away_goals = match.result
            home.register_match(
                scored=home_goals, conceded=away_goals, home=True
            )
            away.register_match(
                scored=away_goals, conceded=home_goals, home=False
            )

            # add the scores to the tournament matrix
            home_points, away_points = self.get_tournament_scores(
                home_goals, away_goals
            )
            self.results_matrix[home.club_id, away.club_id] += home_points
            self.results_matrix[away.club_id, home.club_id] += away_points

    def get_club_names(self, matches):
        seen = {}
        for match in matches:
            for name in (match.home, match.away):
                if name not in seen:
                    seen[name] = True
                    yield name

    def get_tournament_scores(self, home_goals, away_goals):
        """
        Given the results of a match, return the points to assign in the
        results matrix to the home and away teams respectively
        """
        if home_goals > away_goals:
            return (3, 0)
        elif home_goals < away_goals:
            return (0, 3)
        else:
            return (1, 1)

    @property
    def num_clubs(self):
        return len(self.clubs)

class GoalBasedLeague(League):
    """
    Use goals scored/conceded for points in the results matrix
    """
    def get_tournament_scores(self, home_goals, away_goals):
        return (home_goals, away_goals)

class RankingMethod:
    """
    Base class for a method of ranking the clubs in a league
    """
    def rank(self, league: League) -> np.ndarray:
        raise NotImplementedError

    def ordinal_ranking(self, league: League, tie_breakers=None) -> List[Club]:
        tie_breakers = tie_breakers or DEFAULT_TIE_BREAKERS
        ranking_methods = [self] + tie_breakers
        # construct a matrix of scores: entry i,j is the score for club j in
        # ranking method i
        all_scores = np.zeros((len(ranking_methods), league.num_clubs))
        for i, r in enumerate(ranking_methods):
            all_scores[i, :] = r.rank(league)
        return sorted(
            league.clubs,
            key=lambda c: tuple(all_scores[:, c.club_id]),
            reverse=True
        )

class PointsRanking(RankingMethod):
    """
    Rank clubs based on league points
    """
    def rank(self, league):
        return np.array([c.points for c in league.clubs])

class GoalDifferenceRanking(RankingMethod):
    """
    Rank clubs based on goal difference
    """
    def rank(self, league: League):
        return np.array([c.goal_difference for c in league.clubs])

class GoalsForRanking(RankingMethod):
    """
    Rank clubs based on goals scored
    """
    def rank(self, league: League):
        return np.array([c.goals_for for c in league.clubs])

class AveragePointsRanking(RankingMethod):
    """
    Rank by average points across the games played so far
    """
    def rank(self, league):
        return np.array([
            c.points / c.played if c.played > 0 else 0
            for c in league.clubs
        ])

class TournamentRanking(RankingMethod):
    """
    Base class for ranking methods operating solely on the tournament matrix
    """
    def rank(self, league):
        if self.is_reducible(league.results_matrix):
            raise ValueError("tournament matrix is reducible")
        return self._rank(league.results_matrix)

    def _rank(self, results_matrix):
        """
        Return the score vector for the given tournament results matrix.
        """
        raise NotImplementedError

    def get_matches(self, results_matrix):
        """
        Convenience function: return common stuff used in various subclasses.
        Notation is the same as in the González-Díaz paper
        """
        A = results_matrix
        n, _ = A.shape
        M = A + A.T
        m = M @ np.ones((n,))
        return M, m, n

    def get_average_scores(self, A):
        """
        Include average scores as a special case in the base class, since they
        are used in many subclasses
        """
        M, m, n = self.get_matches(A)
        scores = np.zeros((n,))
        for i in range(n):
            scores[i] = sum(A[i, j] / m[i] for j in range(n))
        return scores

    def solve_linear_system(self, a, b, free_val=1):
        """
        Return a solution to a @ x = b
        """
        sol, params = Matrix(a).gauss_jordan_solve(Matrix(b))

        # check that the dimension of the solution space is 0 (unique solution)
        # or 1 (unique solution given a value for the free variable)
        assert len(sol.free_symbols) <= 1, "no unique solution"

        x = sol.xreplace({t: free_val for t in sol.free_symbols})
        return np.array(x).astype("float64").T.flatten()

    @classmethod
    def is_reducible(cls, A):
        """
        Return True iff the square matrix A is reducible
        """
        n, m = A.shape
        assert n == m, "matrix must be square"
        for i in range(n):
            unreachable = {j: True for j in range(n)}
            stack = [i]
            while stack:
                j = stack.pop(-1)
                if j in unreachable:
                    del unreachable[j]
                    for k in range(n):
                        if A[j, k] > 0:
                            stack.append(k)
            if len(unreachable) > 0:
                return True
        return False

class ScoresRanking(TournamentRanking):
    def _rank(self, A):
        return self.get_average_scores(A)

class MaximumLikelihood(TournamentRanking):
    convergence_threshold = 0.000001
    max_iterations = 1000

    def _rank(self, A):
        M, m, n = self.get_matches(A)
        W = A @ np.ones((n,))
        # iterative procedure from 'Solution of a Ranking Problem from Binary
        # Comparisons', L. R Ford (1957), p31, but using the notation from
        # González-Díaz (note that 'A' in Ford is the same as 'M' for
        # González-Díaz...)
        w = np.full((n,), 1 / n)

        for iteration in range(self.max_iterations):
            old_w = np.array(w)
            for i in range(n):
                w[i] = W[i] / sum(M[i, j] / (w[i] + w[j]) for j in range(n))
            w = w / np.sum(w)
            diff = np.max(np.abs(w - old_w))
            if diff <= self.convergence_threshold:
                break
        else:
            raise ValueError(
                "maximum likelihood iteration did not converge"
            )

        x = np.log(w)
        # as per González-Díaz, normalise so x sums to 0 instead of 1
        return x - np.sum(x) / n

class Neustadtl(TournamentRanking):
    def _rank(self, A):
        _, m, _ = self.get_matches(A)
        s = self.get_average_scores(A)
        # divide each row of A by the corresponding number in m
        A_hat = (A.T / m).T
        return A_hat @ s

class FairBets(TournamentRanking):
    def _rank(self, A):
        _, _, n = self.get_matches(A)
        losses = A.T @ np.ones((n,))
        L_inv = np.diag(1 / losses)
        # fair bets returns the unique solution of L_inv * A * x = x, i.e.
        #    x ∈ kernel(L_inv * A - I)
        X = L_inv @ A - np.eye(n)

        w, v = linalg.eig(X)
        kernel_vecs = [i for i in range(n) if np.isclose(w[i], 0)]
        # check that we have actually found the unique solution
        assert len(kernel_vecs) == 1
        x = v[:, kernel_vecs[0]]
        return x / sum(x)

class Buchholz(TournamentRanking):
    def _rank(self, A):
        M, m, _ = self.get_matches(A)
        M_bar = (M.T / m).T
        s = self.get_average_scores(A)
        return M_bar @ s + s

class RecursivePerformance(TournamentRanking):
    def _rank(self, A):
        M, m, n = self.get_matches(A)
        s = self.get_average_scores(A)
        c = -np.log((1 / s) - 1)
        c_hat = c - (np.dot(m, c) / np.sum(m))
        M_bar = (M.T / m).T
        X = M_bar - np.eye(n)
        x = self.solve_linear_system(X, -c_hat)
        const = np.sum(x) / n
        return x - const

class RecursiveBuchholz(TournamentRanking):
    def _rank(self, A):
        M, m, n = self.get_matches(A)
        s = self.get_average_scores(A)
        M_bar = (M.T / m).T
        s_hat = s - 1 / 2
        X = M_bar - np.eye(n)
        x = self.solve_linear_system(X, -s_hat)
        const = np.sum(x) / n
        return x - const

class GeneralisedRowSum(TournamentRanking):
    def _rank(self, A):
        M, m, n = self.get_matches(A)
        m_hat = np.max(M)
        assert n > 2, "need at least three players for generalised row sum"
        eps = 1 / (m_hat * (n - 2))
        A_star = A - A.T
        C = np.diag(m) - M
        s_star = A_star @ np.ones((n,))
        X = np.eye(n) + eps * C
        x = self.solve_linear_system(X, (1 + m_hat * n * eps) * s_star)
        return x / (m_hat * (n - 1))

DEFAULT_TIE_BREAKERS = [GoalDifferenceRanking(), GoalsForRanking()]
