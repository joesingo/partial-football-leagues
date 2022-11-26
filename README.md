# COVID-19 and football: who won the league?

This repo holds the code used in the creation of my short article on ranking
Premier League teams based on the fixtures played so far, in the event that the
remaining fixtures have to be cancelled.

[Read the article here](https://ac.joesingo.co.uk/articles/football/index.html).

## Reproducing the rankings

The main script at `rankings/output.py` runs the various ranking methods:

```
usage: rankings/outputs.py OUTPUT_DIR SUITE [METHOD]
```

At the time of writing, `SUITE` can be `premier` for the partial Premier League
19/20 season, or `intl` for European international fixtures aggregated from the
UEFA Nations 2022 league, 2022 world cup qualifiers, and Euro 2020.

The code provides the most up-to-date documentation on the outputs produced for
each suite.

To set up and run the code:

1. Optionally create a Python virtual environment:
```shell
python3 -m venv venv
. venv/bin/activate
```
2. Install Python dependencies:
```shell
pip install -r requirements.txt
```
3. Run:
```shell
# e.g. premier league
mkdir rankings
python rankings/output.py rankings premier
```

## Adding new data

The results are contained in CSV files in the `data` directory. At the time of
writing, CSVs from [football-data.co.uk](http://www.football-data.co.uk),
[footystats.org](http://www.footystats.org) and
[fixturedownload.com](https://fixturedownload.com/) are supported.

To add support for a new CSV format, create a new subclass of `CSVProvider` in
`rankings/providers.py` with the appropriate CSV header names defined as class
attributes; see the existing classes in that file for examples.

To run the ranking methods on new results data, create a new subclass of
`OutputCreator` in `rankings/output.py` – see `PremierLeagueCovid` and
`InternationalLeague` for examples – and extend the `main()` function as
appropriate.
