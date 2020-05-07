from contextlib import contextmanager
import sys
import inspect
from os import path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rankings import *
from utils import listify, kendall_tau_distance
from conversions import csv_to_fixtures

HERE = path.abspath(path.dirname(__file__))
DATA_PATH = Path(HERE).parent / "data"
CSV_PATH = DATA_PATH / "football-data-co-uk" / "england"
IMAGES_PATH = DATA_PATH / "images"

ABBREVIATIONS = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brighton": "BHA",
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
    "Wolves": "WOL",
}

@contextmanager
def get_fixtures(division: int, year: int):
    y_start = year % 100
    y_end = (y_start + 1) % 100
    year_string = f"{y_start:0>2}{y_end:0>2}"
    csv_path = CSV_PATH / f"{year_string}_e{division}.csv"
    with csv_path.open(encoding="latin_1") as f:
        yield csv_to_fixtures(f)

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
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, func):
        func.output = True
        for k, v in self.kwargs.items():
            setattr(func, k, v)
        return func

class OutputCreator:
    def __init__(self, fixtures, abbrevations=None):
        self.fixtures = fixtures
        self.league = League(self.fixtures, abbrevations)
        self.goal_league = GoalBasedLeague(self.fixtures, abbrevations)

    def run_all(self, outpath):
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if hasattr(method, "output"):
                self.run(name, outpath)

    def run(self, name, outpath):
        method = getattr(self, name)
        if getattr(method, "requires_dir", False):
            outdir = outpath / name
            outdir.mkdir(exist_ok=True)
            method(outdir)
        else:
            ext = getattr(method, "ext", None)
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

        labels = [c.abbrev or c.name for c in league.clubs]
        values = {}
        for i, c1 in enumerate(league.clubs):
            for j, c2 in enumerate(league.clubs):
                score = None
                bg = None
                if c1 == c2:
                    score = "-"
                    bg = 255
                else:
                    raw_score = league.results_matrix[c1.club_id, c2.club_id]
                    score = int(raw_score)
                    relative_score = (
                        (raw_score - min_score) / (max_score - min_score)
                    )
                    bg = 255 - int(relative_score * 255)

                values[(i, j)] = {
                    "bg": f"rgb({bg}, {bg}, {bg})",
                    "text": str(score)
                }
        yield from self.html_matrix(labels, values)

    @output(ext="html")
    def fixtures_so_far(self, outfile):
        labels = [c.abbrev or c.name for c in self.league.clubs]
        values = {}
        for i, c1 in enumerate(self.league.clubs):
            for j, c2 in enumerate(self.league.clubs):
                values[(i, j)] = {
                    "bg": "red" if i != j else "#aaa",
                    "text": ""
                }

        name_to_id = {c.name: c.club_id for c in self.league.clubs}
        for match in self.fixtures.all_matches():
            i = name_to_id[match.home]
            j = name_to_id[match.away]
            values[(i, j)] = {"bg": "white", "text": ""}

        for line in self.html_matrix(labels, values):
            outfile.write(line + "\n")

    def html_matrix(self, labels, values):
        yield ("<style type='text/css'>table{border-collapse:collapse;}td,th{"
               "border:1px solid black;padding:0.7em;text-align:center;}</style>")
        yield "<table>"
        yield "<thead>"
        yield "<tr>"
        yield "<th></th>"
        for label in labels:
            yield f"<th>{label}</th>"
        yield "</tr>"
        yield "</thead>"
        yield "<tbody>"
        for i, label in enumerate(labels):
            yield "<tr>"
            yield f"<th>{label}</th>"
            for j, _ in enumerate(labels):
                value = values[(i, j)]
                if "bg" in value:
                    yield f"<td style='background: {value['bg']}'>"
                else:
                    yield "<td>"
                yield "<span style='background: white'>"
                yield value["text"]
                yield "</span>"
            yield "</tr>"
        yield "</tbody>"
        yield "</table>"

    @listify
    def get_ranking_methods(self):
        for r in all_subclasses(RankingMethod):
            if r not in (TournamentRanking,):
                yield r

    @output()
    def ordinal_rankings_versus_points_ranking(self, _):
        points_ranking = PointsRanking().ordinal_ranking(self.league)

        fig, ax = plt.subplots()

        for r in self.get_ranking_methods():
            ranking = r().ordinal_ranking(self.league)
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

            ax.set_title("Rankings methods versus the current league table")
            ax.set_xlabel("Position in current league table")
            ax.set_ylabel("Position in ranking")
            ticks = np.arange(1, self.league.num_clubs + 1, 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.plot(xs[sort], ys[sort], "o-", label=r.__name__)

        fig.legend()

    @output()
    def scores_versus_points(self, _):
        points = PointsRanking().rank(self.league)
        min_points = np.min(points)
        max_points = np.max(points)

        # work out league ranking (including tie-breakers)
        league_ranking = PointsRanking().ordinal_ranking(self.league)
        perm = np.array([c.club_id for c in league_ranking])

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
            x = xs + i * bar_width
            ax.bar(
                x, rescale(scores[perm], min_points, max_points),
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

    @output(requires_dir=True)
    def historical_league_averages(self, outdir):
        methods = self.get_ranking_methods()

        start_year = 1999
        end_year = 2018
        division = 0
        years = np.arange(start_year, end_year + 1)

        # number of match days in the season is generally around 100. let's
        # take 50 points throughout each season
        num_timesteps = 50
        xs = np.linspace(0, 1, num_timesteps + 1)[1:]

        # load data if it is present, or run the experiment and save
        ys = None
        data = outdir / "data.npy"
        if data.exists():
            print("data already exists: loading...")
            with data.open("rb") as f:
                ys = np.load(f)
        else:
            print("data not found: running experiments...")
            ys = np.zeros((len(years), len(xs), len(methods)))

            for i, year in enumerate(years):
                print(f"{year}-{year + 1} season")
                with get_fixtures(division, year) as fixtures:
                    # get end-of-season results according to normal points
                    # system
                    full_league = League(fixtures)
                    # note: the ranking is a list of club *names*
                    points_ranking = [
                        c.name for c in PointsRanking().ordinal_ranking(
                            full_league
                        )
                    ]
                    club_names = [c.name for c in full_league.clubs]

                    for j, x in enumerate(xs):
                        print(".", end="")
                        sys.stdout.flush()
                        partial_league = League(
                            fixtures.partial(x), club_names=club_names
                        )
                        for k, r in enumerate(methods):
                            try:
                                ranking = [
                                    c.name for c in r().ordinal_ranking(
                                        partial_league,
                                    )
                                ]
                                err = kendall_tau_distance(
                                    ranking,
                                    points_ranking
                                )
                            except ValueError:
                                err = None
                            ys[i, j, k] = err

                    print("")

            print("saving...")
            with data.open("wb") as f:
                np.save(f, ys)

        # plot results
        av_ys = np.mean(ys, axis=0)
        methods_to_plot = (
            PointsRanking, AveragePointsRanking, MaximumLikelihood,
            GeneralisedRowSum, FairBets
        )
        fig, ax = plt.subplots()
        for k, r in enumerate(methods):
            if r in methods_to_plot:
                ax.plot(100 * xs, av_ys[:, k], "o-", label=r.__name__)

        ax.set_title(
            "Average swap distance between partial-season ranking\nand final "
            f"league table ({start_year}-{end_year})"
        )
        ax.set_xlabel("Percentage through season")
        ax.set_ylabel("Average swap distance")
        fig.legend()

    @output()
    def match_days_per_year(self, _):
        start_year = 1999
        end_year = 2018
        # note: the conference league is also available for 2005 onwards, but i
        # will not bother to include it
        divisions = {
            0: "Premier League",
            1: "Championship",
            2: "League 1",
            3: "League 2",
        }

        fig, ax = plt.subplots()
        years = np.arange(start_year, end_year + 1)
        for division, label in divisions.items():
            match_days = np.zeros((len(years),))
            for i, year in enumerate(years):
                with get_fixtures(division, year) as fixtures:
                    match_days[i] = fixtures.num_dates
            ax.plot(years, match_days, label=label)
        fig.legend()
        ax.set_title("Number of match days per season")
        ax.set_xlabel("Season start year")
        ax.set_ylabel("Number of matchdays")
        ax.set_xticks(years[::2])

def main():
    with get_fixtures(0, 2019) as fixtures:
        fc = OutputCreator(fixtures, abbrevations=ABBREVIATIONS)
        outpath = Path("artifacts")
        try:
            name = sys.argv[1]
            fc.run(name, outpath)
        except IndexError:
            fc.run_all(outpath)
        plt.show()

if __name__ == "__main__":
    main()
