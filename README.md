# COVID-19 and football: who won the league?

## Introduction

- Games might have to be cancelled due to COVID-19, but not all fixtures have
  been played
- How should we come up with the final league table?
    - Obvious answer is to just use the current league table. Is this a good
      idea?

    - Consider the following case. Let A, B, C, D be four teams in the league,
      and suppose the only cancelled fixture is B vs C, who are equal on points
      somewhere in the middle of the table. Suppose that A is at the top of the
      league having beaten everyone except B, and D is at the bottom of the
      league having lost to everyone except C.

      Now, since B and C are equal on points and cannot play each other
      directly, using the current leage table would have us rank both teams as
      tied. But is this reasonable?

      Since team A is at the top of the league with only one defeat, it seems
      fair to say that team A is 'good'. Similarly, D is at the bottom with
      only one win, so D can be considered 'bad'. This means B has successfully
      beaten a good team and C has lost to a bad team. Since B and C are
      otherwise equal, it seems only fair to say that B should rank *higher*
      than C in the final ranking.

    - That was an extreme case, but the same considerations apply in more
      modest situations. We should aim for a ranking method which *rewards
      teams for beating good teams*.

- An alternative viewpoint is the following... (**TODO:** make this flow
  nicely)
    - If all teams had played each other, no team can claim to have been
      disadvantaged by the fixture list
    - But with cancellations, a team might be lucky and get out of a
      challenging game, or be unlucky and not get to play an easy game they
      would have most likely won

- What are the solution?
    - League-table derived solutions:
        - Use average points
        - Average points with extra weight for away wins/draws
    - Partial tournament ranking methods:
        - All of those from González-Díaz et. al.

## Analysis

### Ranking methods versus the current league table

- For each tournament ranking method, plot league table position vs rank
- What kind of correlation do we observe?
- One might expect a roughly linear positive correlation
- If there are deviations:
    - Which teams benefit? Which teams are harmed?
    - Conjectures:
        - Those yet to play top teams should benefit
        - Those yet to play poor teams should be harmed
        - Those with higher proportion of home games might benefit
          (if so, can we give extra weight to away goals to counter this?)
- England is not the only country: we could do the same thing for different
  football leagues (and even other sports) to see if the results look the same

### Looking at historical data

- Comparison against the incomplete league table will not tell us which one
  method is 'best'
- After all, if the incomplete table was desirable we would just use that...
- Assuming the (complete) league table system is actually a desirable way to
  rank teams (which seems like a valid assumption since this is the system that
  has been used for years), we can look at historical data
    - Take a league where we know the final standings
    - Run a ranking method on partial leagues at various points (with emphasis
      on the point we are currently at through the Premier League), and see how
      the rankings differ from the known end-of-season rankings
    - We can try to identify those methods that predict the final rankings well

## Implementation notes

- Original results data for 19/20 premier league came from
  [here](https://github.com/openfootball/england/blob/master/2019-20/1-premierleague.txt)

- Results from 1999 to 2020 cam from
  [football-data.co.uk](http://www.football-data.co.uk/englandm.php). The
  script at `data/football-data-co-uk/download_csvs.py` takes the URL to that
  page (or similar pages for other countries besides England) and downloads all
  available CSVs.
