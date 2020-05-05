from pathlib import Path

from bs4 import BeautifulSoup
import requests
import click

BASE_URL = "https://football-data.co.uk"

def get_filename(href):
    """
    Return filename to use for a CSV with the given URL

    e.g. mmz4281/0102/E2.csv -> 0102_e2.csv
    """
    _, season, division = href.split("/")
    return f"{season}_{division.lower()}"

@click.command()
@click.argument("url")
@click.argument("outdir", metavar="DIR")
def main(url: str, outdir: str):
    """
    Find all CSV download links at the given football-data.co.uk URL, and save
    them in the directory given.
    """
    outpath = Path(outdir)
    doc = BeautifulSoup(requests.get(url).content, features="html.parser")
    for a in doc.findAll("a"):
        href = a.get("href")
        if not href.lower().endswith(".csv"):
            continue
        outfile = outpath / get_filename(href)
        print(f"writing to {outfile.name}")
        resp = requests.get(f"{BASE_URL}/{href}")
        outfile.write_bytes(resp.content)

if __name__ == "__main__":
    main()
