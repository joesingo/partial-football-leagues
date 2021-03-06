from csv import DictReader
from datetime import datetime

from rankings import Match, Fixtures
from utils import listify

DATE_FORMATS = ("%d/%m/%y", "%d/%m/%Y")

def parse_date(date_str):
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"date string '{date_str}' does not match any expected date formats"
    )

def csv_to_fixtures(csvfile) -> Fixtures:
    """
    Parse a CSV file from football-data.co.uk and return a Fixtures object

    Note: this assumes that rows in the CSV are sorted in ascending date order
    """
    @listify
    def inner():
        current_date = None
        current_batch = []
        for row in DictReader(csvfile):
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            if not home or not away:
                continue
            date = parse_date(row["Date"])
            if date != current_date:
                current_date = date
                if current_batch:
                    yield current_batch
                    current_batch = []
            result = (int(row["FTHG"]), int(row["FTAG"]))  # full-time home/away goals
            m = Match(home=home, away=away, result=result)
            current_batch.append(m)
        if current_batch:
            yield current_batch
    return Fixtures(inner())
