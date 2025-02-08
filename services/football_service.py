import requests
import logging
from datetime import datetime, timedelta
from config import FOOTBALL_API_BASE_URL, FOOTBALL_API_KEY
from time import sleep
import json
import os
import hashlib

logger = logging.getLogger(__name__)


class FootballService:
    def __init__(self):
        self.base_url = FOOTBALL_API_BASE_URL
        self.headers = {"X-Auth-Token": FOOTBALL_API_KEY}
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
        self.cache_duration = timedelta(hours=1)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.verify_api_access()

    def verify_api_access(self):
        """Verify API key is working"""
        try:
            response = requests.get(
                f"{self.base_url}/competitions/PL",  # Premier League endpoint
                headers=self.headers,
            )
            if response.status_code != 200:
                logger.error(
                    f"API Access Failed: {response.status_code} - {response.text}"
                )
            else:
                logger.info("Football API access verified successfully")
        except Exception as e:
            logger.error(f"API Verification failed: {e}")

    def _get_cache_key(self, endpoint, params=None):
        """Generate a safe cache key from endpoint and params"""
        # Create a string to hash
        key_str = f"{endpoint}_{str(params)}"
        # Create MD5 hash of the key string
        hash_str = hashlib.md5(key_str.encode()).hexdigest()
        return f"cache_{hash_str}.json"

    def _make_request(self, endpoint, params=None, max_retries=3):
        """Make API request with caching and rate limiting"""
        cache_key = self._get_cache_key(endpoint, params)
        cache_file = os.path.join(self.cache_dir, cache_key)

        # Check cache
        if os.path.exists(cache_file):
            modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - modified_time < self.cache_duration:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        logger.info(f"Using cached data for {endpoint}")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading cache: {e}")
                    # Continue to fetch fresh data if cache read fails

        # Rate limiting
        sleep(1)  # Wait 1 second between requests

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}", headers=self.headers, params=params
                )
                response.raise_for_status()
                data = response.json()

                # Cache the response
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception as e:
                    logger.warning(f"Error writing cache: {e}")

                return data
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt + 1 < max_retries:
                    sleep(2**attempt)  # Exponential backoff
                else:
                    # Try to use cached data even if expired
                    if os.path.exists(cache_file):
                        try:
                            with open(cache_file, "r", encoding="utf-8") as f:
                                logger.info(f"Using expired cache for {endpoint}")
                                return json.load(f)
                        except Exception as cache_e:
                            logger.error(f"Error reading expired cache: {cache_e}")
                    raise

    def get_team_stats(self, team_name, timeframe=5):
        """Get team statistics"""
        try:
            # Try all major leagues
            leagues = ["PL", "FL1", "BL1", "SA", "PD"]
            team_data = None

            for league in leagues:
                try:
                    standings_data = self._make_request(
                        f"/competitions/{league}/standings"
                    )
                    # Find team in standings
                    all_teams = []
                    for standing in standings_data["standings"][0]["table"]:
                        team = standing["team"]
                        team["position"] = standing["position"]
                        team["points"] = standing["points"]
                        team["goalsFor"] = standing["goalsFor"]
                        team["goalsAgainst"] = standing["goalsAgainst"]
                        all_teams.append(team)

                    # Try to find team
                    team_data = next(
                        (
                            team
                            for team in all_teams
                            if team["name"].lower() == team_name.lower()
                        ),
                        None,
                    )
                    if team_data:
                        break
                except:
                    continue

            if not team_data:
                logger.warning(f"Team not found in any league: {team_name}")
                return self._get_default_stats()

            team_id = team_data["id"]

            # Get recent matches with specified timeframe
            matches = self._make_request(
                f"/teams/{team_id}/matches",
                params={"limit": timeframe, "status": "FINISHED"},
            )

            if not matches.get("matches"):
                logger.warning(f"No matches found for team: {team_name}")
                return self._get_default_stats()

            # Calculate form from recent matches
            form = self._calculate_form(matches["matches"], team_id)

            # Calculate goals stats
            goals_scored, goals_conceded = self._calculate_goals_stats(
                matches["matches"][-timeframe:], team_id
            )

            # Calculate H2H stats
            h2h_wins = self._calculate_h2h_wins(matches["matches"], team_id)

            # Calculate normalized strength
            max_points = max(t["points"] for t in all_teams)
            normalized_points = team_data["points"] / max_points
            goal_diff = team_data["goalsFor"] - team_data["goalsAgainst"]
            max_goal_diff = max(
                abs(t["goalsFor"] - t["goalsAgainst"]) for t in all_teams
            )
            normalized_goal_diff = (goal_diff + max_goal_diff) / (2 * max_goal_diff)
            strength = normalized_points * 0.7 + normalized_goal_diff * 0.3

            stats = {
                "strength": strength,
                "form": form,
                "head_to_head_wins": h2h_wins,
                "goals_scored_last_5": goals_scored,
                "goals_conceded_last_5": goals_conceded,
                "home_advantage": self._calculate_home_advantage(
                    matches["matches"], team_id
                ),
                "injuries_impact": self._calculate_injuries_impact(
                    self._get_injuries_count(team_id)
                ),
                "league_position": team_data["position"],
            }

            logger.info(f"Stats for {team_name}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting team stats for {team_name}: {str(e)}")
            return self._get_default_stats()

    def get_match_odds(self, team1_id, team2_id):
        """Calculate odds based on team statistics"""
        try:
            # Get team names
            teams = self.get_teams_in_league(2015)  # Ligue 1
            team1_name = next(t["name"] for t in teams if t["id"] == team1_id)
            team2_name = next(t["name"] for t in teams if t["id"] == team2_id)

            # Get team stats
            team1_stats = self.get_team_stats(team1_name)
            team2_stats = self.get_team_stats(team2_name)

            # Calculate power ratings (0-1 scale)
            team1_power = (
                team1_stats["strength"] * 0.4
                + team1_stats["form"] * 0.4
                + (1 - team1_stats["league_position"] / 20) * 0.2
            )

            team2_power = (
                team2_stats["strength"] * 0.4
                + team2_stats["form"] * 0.4
                + (1 - team2_stats["league_position"] / 20) * 0.2
            )

            # Convert to probabilities
            total_power = team1_power + team2_power + 0.15  # 0.15 for draw probability
            home_prob = team1_power / total_power
            away_prob = team2_power / total_power
            draw_prob = 0.15 / total_power

            # Convert probabilities to odds (with margin)
            margin = 1.1  # 10% margin
            home_odds = margin / home_prob
            away_odds = margin / away_prob
            draw_odds = margin / draw_prob

            return {
                "home": round(home_odds, 2),
                "away": round(away_odds, 2),
                "draw": round(draw_odds, 2),
            }

        except Exception as e:
            logger.error(f"Error calculating match odds: {e}")
            return None

    def get_next_match(self, team1_id, team2_id):
        """Get next scheduled match between teams"""
        try:
            # Check all major leagues
            leagues = {
                "PL": "Premier League",
                "SA": "Serie A",
                "PD": "La Liga",
                "BL1": "Bundesliga",
                "FL1": "Ligue 1",
            }

            current_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

            for league_code, league_name in leagues.items():
                try:
                    matches = self._make_request(
                        f"/competitions/{league_code}/matches",
                        params={
                            "dateFrom": current_date,
                            "dateTo": end_date,
                            "status": "SCHEDULED",
                        },
                    )

                    if not matches or "matches" not in matches:
                        continue

                    # Find match with these teams
                    for match in matches["matches"]:
                        home_id = match["homeTeam"].get("id")
                        away_id = match["awayTeam"].get("id")

                        if (home_id == team1_id and away_id == team2_id) or (
                            home_id == team2_id and away_id == team1_id
                        ):
                            match_time = datetime.fromisoformat(
                                match["utcDate"].replace("Z", "+00:00")
                            )
                            venue = match.get("venue", {}).get("name", "")
                            stadium = venue or match["homeTeam"].get("venue", "TBD")

                            logger.info(
                                f"Found match in {league_name}: {match['homeTeam']['name']} vs {match['awayTeam']['name']} at {match_time}"
                            )

                            return {
                                "datetime": match_time,
                                "venue": stadium,
                                "competition": league_name,
                                "home_team": match["homeTeam"]["name"],
                                "away_team": match["awayTeam"]["name"],
                                "matchday": match.get("matchday", "TBD"),
                                "season": match["season"]["id"],
                            }

                except Exception as e:
                    logger.warning(f"Error checking league {league_code}: {e}")
                    continue

            logger.info(
                f"No upcoming match found between team IDs {team1_id} and {team2_id} in any league"
            )
            return None

        except Exception as e:
            logger.error(f"Error getting next match: {e}")
            return None

    def get_available_leagues(self):
        """Get list of available leagues"""
        try:
            response = requests.get(
                f"{self.base_url}/competitions", headers=self.headers
            )
            leagues = response.json().get("competitions", [])
            return [
                {
                    "id": league["id"],
                    "name": league["name"],
                    "country": league["area"]["name"],
                }
                for league in leagues
                if league["type"] == "LEAGUE"  # Only include regular leagues
            ]
        except Exception as e:
            logger.error(f"Error getting leagues: {str(e)}")
            # Return some major leagues as fallback
            return [
                {"id": 2021, "name": "Premier League", "country": "England"},
                {"id": 2014, "name": "La Liga", "country": "Spain"},
                {"id": 2002, "name": "Bundesliga", "country": "Germany"},
                {"id": 2019, "name": "Serie A", "country": "Italy"},
                {"id": 2015, "name": "Ligue 1", "country": "France"},
            ]

    def get_teams_in_league(self, league_id):
        """Get teams in a specific league"""
        try:
            response = requests.get(
                f"{self.base_url}/competitions/{league_id}/teams", headers=self.headers
            )
            teams = response.json().get("teams", [])
            return [
                {
                    "id": team["id"],
                    "name": team["name"],
                    "shortName": team.get("shortName", team["name"]),
                }
                for team in teams
            ]
        except Exception as e:
            logger.error(f"Error getting teams: {str(e)}")
            return []

    def _get_default_stats(self):
        """Return default stats when API fails"""
        return {
            "strength": 0.75,
            "form": 0.5,
            "head_to_head_wins": 0,
            "goals_scored_last_5": 7,
            "goals_conceded_last_5": 5,
            "home_advantage": 0.1,
            "injuries_impact": 0,
            "league_position": 10,
            "odds": {"home": 2.0, "away": 2.0, "draw": 3.0},  # Add default odds
        }

    def _get_team_data(self, team_name):
        """Get basic team information"""
        try:
            # Try exact match first
            response = requests.get(
                f"{self.base_url}/teams",
                headers=self.headers,
                params={"name": team_name},
            )
            data = response.json()

            if not data.get("teams"):
                # Try partial match
                response = requests.get(
                    f"{self.base_url}/teams",
                    headers=self.headers,
                    params={"search": team_name},
                )
                data = response.json()

            if "teams" in data and data["teams"]:
                team = data["teams"][0]
                # Get additional team info
                details_response = requests.get(
                    f"{self.base_url}/teams/{team['id']}", headers=self.headers
                )
                details = details_response.json()

                # Merge team details
                team.update(details)

                # Calculate strength based on multiple factors
                venue_capacity = (
                    float(team.get("venue", {}).get("capacity", 0)) / 100000
                )  # Normalize by 100k
                squad_size = (
                    len(team.get("squad", [])) / 30
                )  # Normalize by typical squad size
                position = team.get("position", 10)

                # Weighted strength calculation
                strength = (
                    (1 - (position / 20)) * 0.4  # League position: 40%
                    + min(venue_capacity, 1) * 0.3  # Stadium size: 30%
                    + min(squad_size, 1) * 0.3  # Squad depth: 30%
                )

                team["strength"] = min(max(strength, 0.3), 1.0)
                logger.info(
                    f"Calculated strength for {team_name}: {team['strength']:.2f}"
                )
                return team

            logger.warning(f"No team found for name: {team_name}")
            return None

        except Exception as e:
            logger.error(f"Error getting team data for {team_name}: {str(e)}")
            return None

    def _get_recent_matches(self, team_id, limit=5):
        """Get recent matches for a team"""
        endpoint = f"/teams/{team_id}/matches"
        response = requests.get(
            f"{self.base_url}{endpoint}", headers=self.headers, params={"limit": limit}
        )
        return response.json().get("matches", [])

    def _calculate_goals_stats(self, matches, team_id):
        """Calculate goals scored and conceded in last 5 matches"""
        scored = 0
        conceded = 0
        for match in matches:
            if match["homeTeam"]["id"] == team_id:
                scored += match["score"]["fullTime"]["home"] or 0
                conceded += match["score"]["fullTime"]["away"] or 0
            else:
                scored += match["score"]["fullTime"]["away"] or 0
                conceded += match["score"]["fullTime"]["home"] or 0
        return scored, conceded

    def _calculate_h2h_wins(self, matches, team_id):
        """Calculate head-to-head wins"""
        wins = 0
        for match in matches:
            if (
                match["homeTeam"]["id"] == team_id
                and match["score"]["winner"] == "HOME_TEAM"
            ) or (
                match["awayTeam"]["id"] == team_id
                and match["score"]["winner"] == "AWAY_TEAM"
            ):
                wins += 1
        return wins

    def _calculate_home_advantage(self, matches, team_id):
        """Calculate home advantage based on home vs away performance"""
        home_points = 0
        away_points = 0
        home_games = 0
        away_games = 0

        for match in matches:
            if match["homeTeam"]["id"] == team_id:
                home_games += 1
                if match["score"]["winner"] == "HOME_TEAM":
                    home_points += 3
                elif match["score"]["winner"] == "DRAW":
                    home_points += 1
            else:
                away_games += 1
                if match["score"]["winner"] == "AWAY_TEAM":
                    away_points += 3
                elif match["score"]["winner"] == "DRAW":
                    away_points += 1

        home_ppg = home_points / max(home_games, 1)
        away_ppg = away_points / max(away_games, 1)
        return (home_ppg - away_ppg) / 3  # Normalize to [-1, 1]

    def _get_league_position(self, team_id):
        """Get team's current league position"""
        try:
            response = requests.get(
                f"{self.base_url}/teams/{team_id}", headers=self.headers
            )
            data = response.json()
            return data.get("position", 10)
        except:
            return 10

    def _get_injuries_count(self, team_id):
        """Get number of injured players"""
        try:
            response = requests.get(
                f"{self.base_url}/teams/{team_id}/injuries", headers=self.headers
            )
            data = response.json()
            return len(data.get("injuries", []))
        except:
            return 0

    def _calculate_injuries_impact(self, injuries_count):
        """Calculate impact of injuries on team performance"""
        return min(injuries_count * 0.1, 1.0)  # Cap at 1.0

    def _calculate_form(self, matches, team_id):
        """Calculate form based on recent results"""
        try:
            points = []
            for match in matches[-5:]:  # Last 5 matches
                if match["score"]["winner"] == "DRAW":
                    points.append(1)
                elif (
                    match["homeTeam"]["id"] == team_id
                    and match["score"]["winner"] == "HOME_TEAM"
                ) or (
                    match["awayTeam"]["id"] == team_id
                    and match["score"]["winner"] == "AWAY_TEAM"
                ):
                    points.append(3)
                else:
                    points.append(0)

            # Weight recent matches more heavily
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(points) :]
            weighted_form = sum(p * w for p, w in zip(points, weights)) / sum(weights)
            return weighted_form / 3  # Normalize to 0-1 scale

        except Exception as e:
            logger.error(f"Error calculating form: {e}")
            return 0.5
