from csv import DictReader

from rankings import Match

def csv_to_matches(csvfile):
    """
    Parse a CSV file from football-data.co.uk and return Match objects
    """
    for row in DictReader(csvfile):
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        if not home or not away:
            continue
        result = (int(row["FTHG"]), int(row["FTAG"]))  # full-time home/away goals
        yield Match(home=home, away=away, result=result)
