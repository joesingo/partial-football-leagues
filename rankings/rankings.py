from dataclasses import dataclass
from typing import Dict
import itertools

import numpy as np
import numpy.linalg as linalg
from sympy import Matrix

@dataclass
class Club:
    """
    Representation of a club in a league
    """
    name: str
    club_id: int

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
    clubs: [Club]
    club_ids: Dict[str, int]

    results_matrix: np.ndarray

    def __init__(self, results):
        self.clubs = []
        self.club_ids = {}

        # init clubs
        for name in self.get_club_names(results):
            club_id = len(self.clubs)
            club = Club(name=name, club_id=club_id)
            self.clubs.append(club)
            self.club_ids[name] = club_id

        # create tournament matrix
        self.results_matrix = np.zeros((self.num_clubs, self.num_clubs))
        for match_dict in results:
            home = self.clubs[self.club_ids[match_dict["home"]]]
            away = self.clubs[self.club_ids[match_dict["away"]]]
            scores = match_dict["result"]

            # record the match details in the Club objects
            home_goals, away_goals = scores
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

    def get_club_names(self, results):
        seen = {}
        for match in results:
            for name in (match["home"], match["away"]):
                if name not in seen:
                    yield name
                    seen[name] = True
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
    def rank(self, league: League) -> [Club]:
        raise NotImplementedError

class PointsRanking(RankingMethod):
    """
    Rank clubs based on league points
    """
    def rank(self, league):
        return sorted(
            league.clubs,
            key=lambda c: (c.points, c.goal_difference, c.goals_for),
            reverse=True
        )

class AveragePointsRanking(RankingMethod):
    """
    Rank by average points across the games played so far
    """
    def rank(self, league):
        return sorted(
            league.clubs,
            key=lambda c: (
                c.points / c.played,
                c.goal_difference,
                c.goals_for / c.played
            ),
            reverse=True
        )

class TournamentRanking(RankingMethod):
    """
    Base class for ranking methods operating solely on the tournament matrix
    """
    def rank(self, league):
        scores = self._rank(league.results_matrix)
        return sorted(
            league.clubs,
            key=lambda c: scores[c.club_id],
            reverse=True
        )

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
