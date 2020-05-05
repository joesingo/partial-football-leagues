import sys
import inspect
from os import path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rankings import (
    League,
    GoalBasedLeague,
    RankingMethod,
    PointsRanking,
    AveragePointsRanking,
    TournamentRanking,
    MaximumLikelihood,
    FairBets,
    GeneralisedRowSum,
)

from conversions import csv_to_matches

HERE = path.abspath(path.dirname(__file__))
RESULTS_PATH = Path(HERE).parent / "data" / "football-data-co-uk" / "england" / "1920_e0.csv"

ABBREVIATIONS = {
    "Arsenal": "ARS",
    "Aston Villa": "AVA",
    "Bournemouth": "BOU",
    "Brighton": "BRH",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Man City": "MCI",
    "Man United": "MUN",
    "Newcastle": "NEW",
    "Norwich": "NOR",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Tottenham": "TOT",
    "Watford": "WAT",
    "West Ham": "WHU",
    "Wolves": "WLV",
}

def all_subclasses(cls):
    for child in cls.__subclasses__():
        yield child
        yield from all_subclasses(child)

def rescale(xs, min_val, max_val):
    """
    scale xs so that it's min and max entries are as given
    """
    min_x = np.min(xs)
    max_x = np.max(xs)
    normed = (xs - min_x) / (max_x - min_x)
    return min_val + (max_val - min_val) * normed

class output:
    """
    Decorator to mark a method as being a output generation method
    """
    def __init__(self, ext=None):
        self.ext = ext

    def __call__(self, func):
        func.output = True
        func.ext = self.ext
        return func

class OutputCreator:
    def __init__(self, matches, abbrevations=None):
        self.league = League(matches, abbrevations)
        self.goal_league = GoalBasedLeague(matches, abbrevations)

    def run_all(self, outpath):
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if hasattr(method, "output"):
                self.run(name, outpath)

    def run(self, name, outpath):
        method = getattr(self, name)
        ext = method.ext
        filename = f"{name}.{ext}" if ext else name
        p = outpath / filename
        with p.open("w") as outfile:
            method(outfile)

    @output(ext="html")
    def tournament_points_based(self, outfile):
        for line in self.tournament_table(self.league):
            outfile.write(line + "\n")

    @output(ext="html")
    def tournament_goals_based(self, outfile):
        for line in self.tournament_table(self.goal_league):
            outfile.write(line)

    def tournament_table(self, league):
        n = league.num_clubs
        min_score = min(
            league.results_matrix[i, j]
            for i in range(n) for j in range(n) if j != i
        )
        max_score = max(
            league.results_matrix[i, j]
            for i in range(n) for j in range(n) if j != i
        )

        yield ("<style type='text/css'>table{border-collapse:collapse;}td,th{"
               "border:1px solid black;padding:0.7em;text-align:center;}</style>")
        yield "<table>"
        yield "<thead>"
        yield "<tr>"
        yield "<th></th>"
        for club in league.clubs:
            yield f"<th>{club.abbrev or club.name}</th>"
        yield "</tr>"
        yield "</thead>"
        yield "<tbody>"
        for c1 in league.clubs:
            yield "<tr>"
            yield f"<th>{c1.abbrev or c1.name}</th>"
            for c2 in league.clubs:
                score = None
                bg = None
                fg = None
                if c1 == c2:
                    score = "-"
                    bg = 255
                    fg = 0
                else:
                    raw_score = league.results_matrix[c1.club_id, c2.club_id]
                    score = int(raw_score)
                    relative_score = (
                        (raw_score - min_score) / (max_score - min_score)
                    )
                    bg = 255 - int(relative_score * 255)
                    fg = 255 - bg

                bg_colour = f"rgb({bg}, {bg}, {bg})"
                fg_colour = f"rgb({fg}, {fg}, {fg})"
                yield (f"<td style='background: {bg_colour}'>"
                       f"<span style='background:white'>{score}</span></td>")

            yield "</tr>"
        yield "</tbody>"
        yield "</table>"

    def get_ranking_methods(self):
        for r in all_subclasses(RankingMethod):
            if r not in (TournamentRanking,):
                yield r

    @output()
    def ordinal_rankings_versus_points_ranking(self, _):
        points = PointsRanking().rank(self.league)
        points_ranking = RankingMethod.ordinal_ranking(points)

        for r in self.get_ranking_methods():
            scores = r().rank(self.league)
            ranking = RankingMethod.ordinal_ranking(scores)
            print(r.__name__)

            xs = []
            ys = []

            for pos, club in enumerate(ranking):
                pt_pos = points_ranking.index(club)
                diff_str = ""
                diff = pt_pos - pos
                if diff > 0:
                    diff_str = f"(+{diff})"
                elif diff < 0:
                    diff_str = f"(-{-diff})"
                print(f"{1 + pos}. {club.name} {diff_str}")

                xs.append(1 + pt_pos)
                ys.append(1 + pos)

            print("")

            # we want to display points-based positions on the horizontal axis,
            # but the xs and ys were built in ranking-based order. need to sort
            # them
            xs = np.array(xs)
            ys = np.array(ys)
            sort = np.argsort(xs)

            plt.title("Rankings methods versus the current league table")
            plt.xlabel("Position in current league table")
            plt.ylabel("Position in ranking")
            ticks = np.arange(1, self.league.num_clubs + 1, 1)
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.plot(xs[sort], ys[sort], "o-", label=r.__name__)

        plt.legend()

    @output()
    def scores_versus_points(self, _):
        points = PointsRanking().rank(self.league)
        points_vector = np.array([s for c, s in points])
        min_points = np.min(points_vector)
        max_points = np.max(points_vector)
        perm = np.argsort(points_vector)[::-1]  # reverse to sort descending

        ranking_methods = [
            PointsRanking,
            AveragePointsRanking,
            MaximumLikelihood,
            FairBets,
            GeneralisedRowSum
        ]
        bar_width = 0.9 / len(ranking_methods)
        xs = np.arange(len(self.league.clubs))
        fig, ax = plt.subplots()

        for i, r in enumerate(ranking_methods):
            scores = r().rank(self.league)
            scores_vector = np.array([s for c, s in scores])
            x = xs + i * bar_width
            ax.bar(
                x, rescale(scores_vector[perm], min_points, max_points),
                bar_width, align="edge", label=r.__name__
            )

        ax.set_title("Comparison of scores between ranking methods")
        ax.set_xlabel("Club")
        ax.set_ylabel("Score (scaled for comparison against points)")
        ax.set_xticks(xs + bar_width * len(ranking_methods) / 2)
        club_names = np.array([c.abbrev or c.name for c in self.league.clubs])
        ax.set_xticklabels(club_names[perm])
        ax.legend()
        fig.tight_layout()


def main():
    with open(RESULTS_PATH) as f:
        matches = list(csv_to_matches(f))
        fc = OutputCreator(matches, ABBREVIATIONS)
        outpath = Path("/tmp/f")
        try:
            name = sys.argv[1]
            fc.run(name, outpath)
        except IndexError:
            fc.run_all(outpath)
        plt.show()

if __name__ == "__main__":
    main()
