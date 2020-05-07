from pathlib import Path

import click
from bs4 import BeautifulSoup
import requests

URL = "https://www.premierleague.com/tables"

@click.command()
@click.argument("outdir", metavar="DIR")
def main(outdir: str):
    """
    Download badges of premier league teams and save in the given directory.
    """
    outpath = Path(outdir)
    doc = BeautifulSoup(requests.get(URL).content, features="html.parser")
    container = doc.find("div", attrs={"class": "tableContainer"})
    for td in container.find_all("td", attrs={"class": "team"}):
        abbrev = td.find("span", attrs={"class": "short"}).text
        img_url = td.find("img").get("src")
        outfile = outpath / f"{abbrev.lower()}.png"
        print(f"writing to {outfile.name}")
        resp = requests.get(img_url)
        outfile.write_bytes(resp.content)

if __name__ == "__main__":
    main()
