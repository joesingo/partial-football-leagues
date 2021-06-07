from csv import DictReader
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import List

from rankings import Match, Fixtures
from utils import listify

class DateFormatType(Enum):
    UNIX_TIMESTAMP = 1
    STRPTIME = 2

class CSVProvider:
    # to be overridden in child classes
    date_field = None
    date_format_type = None
    date_formats = None
    home_team_name_field = None
    away_team_name_field = None
    home_team_goals_field = None
    away_team_goals_field = None

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
                if not self.row_is_valid(row):
                    continue
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
        if self.date_format_type == DateFormatType.UNIX_TIMESTAMP:
            try:
                timestamp = int(date_str)
            except ValueError:
                raise ValueError(
                    f"invalid timestamp '{date_str}'"
                )
            return datetime.utcfromtimestamp(timestamp)

        elif self.date_format_type == DateFormatType.STRPTIME:
            for fmt in self.date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(
                f"date string '{date_str}' does not match any expected date formats"
            )

        raise ValueError(
            f"unknown date format type: '{self.date_format_type}'"
        )

    def row_is_valid(self, row: dict) -> bool:
        """
        Returns True if row is valid and False otherwise
        """
        return True

class FootballDataProvider(CSVProvider):
    """
    For CSV data from football-data.co.uk
    """
    date_field = "Date"
    date_format_type = DateFormatType.STRPTIME
    date_formats = ("%d/%m/%y", "%d/%m/%Y")
    home_team_name_field = "HomeTeam"
    away_team_name_field =  "AwayTeam"
    home_team_goals_field = "FTHG"
    away_team_goals_field = "FTAG"

class FootyStatsProvider(CSVProvider):
    date_field = "timestamp"
    date_format_type = DateFormatType.UNIX_TIMESTAMP
    date_formats = []
    home_team_name_field = "home_team_name"
    away_team_name_field = "away_team_name"
    home_team_goals_field = "home_team_goal_count"
    away_team_goals_field = "away_team_goal_count"

    def row_is_valid(self, row):
        return row["status"] == "complete"
