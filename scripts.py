import requests
import pandas as pd

# Get all teams
url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"
teams = []

for i in range(1, 20):  # There are ~360 teams, paginated in batches of 50 or so
    r = requests.get(f"{url}?limit=50&offset={(i - 1) * 50}")
    data = r.json()
    teams.extend(data["sports"][0]["leagues"][0]["teams"])
    if not data["sports"][0]["leagues"][0]["teams"]:
        break  # No more teams

# Extract IDs and names
team_info = [
    {
        "id": team["team"]["id"],
        "name": team["team"]["displayName"],
        "abbreviation": team["team"]["abbreviation"]
    }
    for team in teams
]

df = pd.DataFrame(team_info)
print(df.head())
