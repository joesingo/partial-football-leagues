from csv import DictReader
from datetime import datetime
from dataclasses import dataclass
from typing import List

from rankings import Match, Fixtures
from utils import listify

@dataclass
class CSVProvider:
    date_field: str
    date_formats: List[str]
    home_team_name_field: str
    away_team_name_field: str
    home_team_goals_field: str
    away_team_goals_field: str

    def csv_to_fixtures(self, csvfile) -> Fixtures:
        """
        Parse a CSV file and return a Fixtures object

        Note: this assumes that rows in the CSV are sorted in ascending date order
        """
        @listify
        def inner():
            current_date = None
            current_batch = []
            for row in DictReader(csvfile):
                home = row[self.home_team_name_field]
                away = row[self.away_team_name_field]
                if not home or not away:
                    continue
                date = self.parse_date(row[self.date_field])
                if date != current_date:
                    current_date = date
                    if current_batch:
                        yield current_batch
                        current_batch = []
                home_goals = int(row[self.home_team_goals_field])
                away_goals = int(row[self.away_team_goals_field])
                result = (home_goals, away_goals)
                m = Match(home=home, away=away, result=result)
                current_batch.append(m)
            if current_batch:
                yield current_batch
        return Fixtures(inner())

    def parse_date(self, date_str: str) -> datetime:
        for fmt in self.date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(
            f"date string '{date_str}' does not match any expected date formats"
        )

football_data_provider = CSVProvider(
    "Date",
    ("%d/%m/%y", "%d/%m/%Y"),
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG"
)
