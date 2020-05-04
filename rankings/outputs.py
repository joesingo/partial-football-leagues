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

def output(func):
    """
    Decorator to mark a method as being a output generation method
    """
    func.output = True
    return func

class OutputCreator:
    def __init__(self, league):
        self.league = league

    def run_all(self):
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if not hasattr(method, "output"):
                continue

            method()

    @output
    def home_versus_away_games(self):
        print("played vs home vs away games:")
        for club in self.league.clubs:
            home = club.home_games
            away = club.played - home
            print(f"{club.name}: {club.played}, {home}, {away}")

    @output
    def rankings_versus_points_ranking(self):
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
        l = League(results)
        fc = OutputCreator(l)
        fc.run_all()

if __name__ == "__main__":
    main()
