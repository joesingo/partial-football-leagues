import inspect
import json
from os import path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rankings import (
    League,
    GoalBasedLeague,
    RankingMethod,
    PointsRanking,
    TournamentRanking,
)

HERE = path.abspath(path.dirname(__file__))
RESULTS_PATH = Path(HERE).parent / "data" / "results.json"

def all_subclasses(cls):
    for child in cls.__subclasses__():
        yield child
        yield from all_subclasses(child)

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
    def __init__(self, results):
        self.league = League(results)
        self.goal_league = GoalBasedLeague(results)

    def run_all(self, outpath):
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if not hasattr(method, "output"):
                continue
            ext = method.ext
            filename = f"{name}.{ext}" if ext else name
            p = outpath / filename
            with p.open("w") as outfile:
                method(outfile)

    @output(ext="html")
    def tournament_points_based(self, outfile):
        for line in self.tournament_table(self.league):
            outfile.write(line)

    @output(ext="html")
    def tournament_goals_based(self, outfile):
        for line in self.tournament_table(self.goal_league):
            outfile.write(line)

    def tournament_table(self, league):
        yield ("<style type='text/css'>table{border-collapse:collapse;}td{"
               "border:1px solid black;}</style>")
        yield "<table>"
        yield "<thead>"
        yield "<tr>"
        yield "<th></th>"
        for club in league.clubs:
            yield f"<th>{club.name}</th>"
        yield "</tr>"
        yield "</thead>"
        yield "<tbody>"
        for c1 in league.clubs:
            yield "<tr>"
            yield f"<th>{c1.name}</th>"
            for c2 in league.clubs:
                score = int(league.results_matrix[c1.club_id, c2.club_id])
                yield f"<td>{score}</td>"
            yield "</tr>"
        yield "</tbody>"
        yield "</table>"

    @output()
    def rankings_versus_points_ranking(self, _):
        return
        points_ranking = PointsRanking().rank(self.league)

        for r in all_subclasses(RankingMethod):
            if r in (PointsRanking, TournamentRanking):
                continue
            ranking = r().rank(self.league)
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
        plt.show()

def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
        fc = OutputCreator(results)
        fc.run_all(Path("/tmp/f"))

if __name__ == "__main__":
    main()
