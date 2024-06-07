from contextlib import contextmanager
import sys
import inspect
from os import path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rankings import (
    PointsRanking, AveragePointsRanking, MaximumLikelihood, Neustadtl,
    RecursivePerformance, FairBets, League, GoalBasedLeague, RankingMethod,
    GoalDifferenceRanking, GoalsForRanking, Fixtures,
)
from utils import kendall_tau_distance
from providers import (
    FootballDataProvider, FootyStatsProvider, FixturedownloadProvider
)
from abbreviations import INTL_ABBREVIATIONS, PREMIER_ABBREVIATIONS

HERE = path.abspath(path.dirname(__file__))
DATA_PATH = Path(HERE).parent / "data"
PREMIER_CSV_PATH = DATA_PATH / "football-data-co-uk" / "england"
FOOTYSTATS_PATH = DATA_PATH / "footystats-org"
FIXTUREDOWNLOAD_PATH = DATA_PATH / "fixturedownload.com"
IMAGES_PATH = DATA_PATH / "images"

@contextmanager
def get_premier_fixtures(division: int, year: int):
    y_start = year % 100
    y_end = (y_start + 1) % 100
    year_string = f"{y_start:0>2}{y_end:0>2}"
    csv_path = PREMIER_CSV_PATH / f"{year_string}_e{division}.csv"
    with csv_path.open(encoding="latin_1") as f:
        yield FootballDataProvider().csv_to_fixtures(f)


@contextmanager
def get_footystats_fixtures(csv_path):
    with csv_path.open() as f:
        yield FootyStatsProvider().csv_to_fixtures(f)

@contextmanager
def get_fixturedownload_fixtures(csv_path):
    with csv_path.open() as f:
        yield FixturedownloadProvider().csv_to_fixtures(f)


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
    special_methods = [
        PointsRanking,
        AveragePointsRanking,
        MaximumLikelihood,
        Neustadtl,
        RecursivePerformance,
        FairBets
    ]
    require_irreducible = True

    def __init__(self):
        pass

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
            mode = "wb" if getattr(method, "binary", False) else "w"
            with p.open(mode) as outfile:
                method(outfile)

    @output(ext="html")
    def ordinal_ranking_table_points_based(self, outfile):
        self.ordinal_ranking_helper(outfile, self.league)

    @output(ext="html")
    def ordinal_ranking_table_goal_based(self, outfile):
        self.ordinal_ranking_helper(outfile, self.goal_league)

    def ordinal_ranking_helper(self, outfile, league):
        r_methods = self.special_methods
        rankings = [
            r().ordinal_ranking(league,
                                require_irreducible=self.require_irreducible)
            for r in r_methods
        ]
        points_ranks = {c.name: i for i, c in enumerate(rankings[0])}

        def get_html_lines():
            # todo: make a new method to wrap creation of HTML table?
            yield ("<style type='text/css'>table{border-collapse:collapse;}td,th{"
                   "border:1px solid black;padding:0.5em;text-align:center;}</style>")
            yield "<table class='ordinal_rankings'>"
            yield "<thead>"
            yield "<tr>"
            yield "<th></th>"
            for r in r_methods:
                yield f"<th>{r.display_name}</th>"
            yield "</tr>"
            yield "</thead>"
            yield "<tbody>"

            for i, clubs in enumerate(zip(*rankings)):
                yield "<tr>"
                yield f"<th>{i + 1}</th>"
                for club in clubs:
                    yield "<td>"
                    diff = points_ranks[club.name] - i
                    display_name = club.abbrev or club.name
                    if diff != 0:
                        yield f"<b>{display_name}</b>"
                    else:
                        yield display_name
                    if diff > 0:
                        yield f" <span style='color: green'>(+{diff})</span>"
                    elif diff < 0:
                        yield f" <span style='color: red'>(-{-diff})</span>"
                    yield "</td>"
                yield "</tr>"
            yield "</tbody>"
            yield "</table>"

        for line in get_html_lines():
            outfile.write(line + "\n")

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

    def html_matrix(self, labels, values):
        yield ("<style type='text/css'>table{border-collapse:collapse;}td,th{"
               "border:1px solid black;padding:0.7em;text-align:center;}</style>")
        yield "<table class='matrix'>"
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


class PremierLeagueCovid(OutputCreator):
    def __init__(self, fixtures):
        super().__init__()
        self.fixtures = fixtures
        self.league = League(self.fixtures, PREMIER_ABBREVIATIONS)
        self.goal_league = GoalBasedLeague(
            self.fixtures, PREMIER_ABBREVIATIONS)

    @output(ext="html")
    def tournament_points_based(self, outfile):
        for line in self.tournament_table(self.league):
            outfile.write(line + "\n")

    @output(ext="html")
    def tournament_goals_based(self, outfile):
        for line in self.tournament_table(self.goal_league):
            outfile.write(line)

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
        for match in self.fixtures.matches:
            i = name_to_id[match.home]
            j = name_to_id[match.away]
            values[(i, j)] = {"bg": "white", "text": ""}

        for line in self.html_matrix(labels, values):
            outfile.write(line + "\n")

    @output()
    def ordinal_rankings_versus_points_ranking(self, _):
        points_ranking = PointsRanking().ordinal_ranking(self.league)

        fig, ax = plt.subplots()

        for r in self.special_methods:
            ranking = r().ordinal_ranking(self.league)
            print(r.display_name)

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
            ax.plot(xs[sort], ys[sort], "o-", label=r.display_name)

        fig.legend()

    @output(binary=True, ext="png")
    def scores_versus_points(self, outfile):
        points = PointsRanking().rank(self.league)
        min_points = np.min(points)
        max_points = np.max(points)

        # work out league ranking (including tie-breakers)
        league_ranking = PointsRanking().ordinal_ranking(self.league)
        perm = np.array([c.club_id for c in league_ranking])

        ranking_methods = self.special_methods
        bar_width = 0.9 / len(ranking_methods)
        xs = np.arange(len(self.league.clubs))
        fig, ax = plt.subplots()

        for i, r in enumerate(ranking_methods):
            scores = r().rank(self.league)
            x = xs + i * bar_width
            ax.bar(
                x, rescale(scores[perm], min_points, max_points),
                bar_width, align="edge", label=r.display_name
            )

        ax.set_title("Comparison of scores between ranking methods")
        ax.set_xlabel("Club")
        ax.set_ylabel("Score (scaled for comparison against league points)")
        ax.set_xticks(xs + bar_width * len(ranking_methods) / 2)
        club_names = np.array([c.abbrev or c.name for c in self.league.clubs])
        ax.set_xticklabels(club_names[perm])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outfile, format="png")

    @output(requires_dir=True)
    def historical_league_averages(self, outdir):
        methods = RankingMethod.get_ranking_methods()

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
                with get_premier_fixtures(division, year) as fixtures:
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
        methods_to_plot = self.special_methods
        fig, ax = plt.subplots()
        for k, r in enumerate(methods):
            if r in methods_to_plot and r not in (GoalDifferenceRanking, GoalsForRanking):
                ax.plot(100 * xs, av_ys[:, k], "o-", label=r.display_name)

        ax.set_title(
            "Average swap distance between partial-season ranking\nand final "
            f"league table ({start_year}-{end_year})"
        )
        ax.set_xlabel("Percentage through season")
        ax.set_ylabel("Average swap distance")
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.8))
        fig.tight_layout()
        with (outdir / "graph.png").open("wb") as f:
            fig.savefig(f, format="png")

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
                with get_premier_fixtures(division, year) as fixtures:
                    match_days[i] = fixtures.num_dates
            ax.plot(years, match_days, label=label)
        fig.legend()
        ax.set_title("Number of match days per season")
        ax.set_xlabel("Season start year")
        ax.set_ylabel("Number of matchdays")
        ax.set_xticks(years[::2])


class InternationalLeague(OutputCreator):
    require_irreducible = False

    def __init__(self, fixture_list):
        self.fixtures = Fixtures.merge(fixture_list)
        self.league = League(self.fixtures, INTL_ABBREVIATIONS)
        self.goal_league = GoalBasedLeague(self.fixtures, INTL_ABBREVIATIONS)

        for batch in self.fixtures.matches_by_date:
            date = batch[0].date
            d = date.strftime("%d-%m-%Y")
            print(f"{d}:")
            for match in batch:
                hg, ag = match.result
                print(f"\t{match.home} vs {match.away}: {hg} - {ag}")

    @output(ext="html")
    def tournament_points_based(self, outfile):
        for line in self.tournament_table(self.league):
            outfile.write(line + "\n")

    @output(ext="html")
    def tournament_goals_based(self, outfile):
        for line in self.tournament_table(self.goal_league):
            outfile.write(line)


def usage():
    print(f"usage: {sys.argv[0]} OUTPUT_DIR SUITE [METHOD]", file=sys.stderr)


def main():
    try:
        output_dir = sys.argv[1]
        suite = sys.argv[2]
    except IndexError:
        usage()
        sys.exit(1)
    try:
        method = sys.argv[3]
    except IndexError:
        method = None

    outpath = Path(output_dir)

    if suite == "premier":
        with get_premier_fixtures(0, 2019) as fixtures:
            fc = PremierLeagueCovid(fixtures)
            if method is not None:
                fc.run(method, outpath)
            else:
                fc.run_all(outpath)

    elif suite == "intl":
        uefa_path = FOOTYSTATS_PATH / "uefa-nations-2022.csv"
        wc_path = FOOTYSTATS_PATH / "world-cup-europe-quals-2022.csv"
        euro_path = FIXTUREDOWNLOAD_PATH / "euro-2020.csv"
        g = get_footystats_fixtures
        h = get_fixturedownload_fixtures
        with g(uefa_path) as uefa, g(wc_path) as wc, h(euro_path) as euros:
            fc = InternationalLeague([uefa, wc, euros])
            if method is not None:
                fc.run(method, outpath)
            else:
                fc.run_all(outpath)

    elif suite == "intl2":
        uefa_path = FOOTYSTATS_PATH / "uefa-nations-22-23.csv"
        euro_path = FOOTYSTATS_PATH / "euro-quals-22-24.csv"
        g = get_footystats_fixtures
        h = get_fixturedownload_fixtures
        with g(uefa_path) as uefa, g(euro_path) as euro:
            fc = InternationalLeague([uefa, euro])
            if method is not None:
                fc.run(method, outpath)
            else:
                fc.run_all(outpath)

    else:
        print(f"unknown suite '{suite}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
