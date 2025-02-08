from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from services.prediction_service import PredictionService
from services.football_service import FootballService
import sys
from utils.keyboard_helper import get_timeframe_keyboard, get_statistics_keyboard
import matplotlib.pyplot as plt
import io
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

prediction_service = PredictionService()
football_service = FootballService()

# Store selected league/team data
user_selections = {}

ADMIN_USER_IDS = [
    1273086972,  # Replace this with the ID you got from @userinfobot
]


class UserState:
    def __init__(self):
        self.league_id = None
        self.first_team = None
        self.second_team = None
        self.timeframe = 5


user_states = {}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available leagues"""
    leagues = football_service.get_available_leagues()

    # Create keyboard with league buttons
    keyboard = []
    for league in leagues:
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{league['name']} ({league['country']})",
                    callback_data=f"league_{league['id']}",
                )
            ]
        )

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Welcome to BetWinner! 🎯\nPlease select a league:", reply_markup=reply_markup
    )


class ButtonHandlers:
    @staticmethod
    async def handle_h2h(team1_id, team2_id, teams, context, chat_id):
        """Handle head-to-head button"""
        team1_name = next(t["name"] for t in teams if t["id"] == team1_id)
        team2_name = next(t["name"] for t in teams if t["id"] == team2_id)

        # Get H2H matches
        matches = football_service._make_request(
            f"/teams/{team1_id}/matches", params={"limit": 50, "status": "FINISHED"}
        )

        # Filter H2H matches
        h2h_matches = [
            m
            for m in matches.get("matches", [])
            if (m["homeTeam"]["id"] == team2_id or m["awayTeam"]["id"] == team2_id)
        ]

        if not h2h_matches:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"No head-to-head matches found between {team1_name} and {team2_name}",
            )
            return

        # Format H2H results
        h2h_text = [f"🤝 Head-to-Head: {team1_name} vs {team2_name}"]
        for match in h2h_matches[:5]:  # Show last 5 H2H matches
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            score = f"{match['score']['fullTime']['home']}-{match['score']['fullTime']['away']}"
            date = match["utcDate"].split("T")[0]
            h2h_text.append(f"• {date}: {home} {score} {away}")

        await context.bot.send_message(chat_id=chat_id, text="\n".join(h2h_text))

    @staticmethod
    async def handle_form(team1_id, team2_id, teams, context, chat_id):
        """Handle form guide button"""
        team1_name = next(t["name"] for t in teams if t["id"] == team1_id)
        team2_name = next(t["name"] for t in teams if t["id"] == team2_id)

        form_text = ["📊 Recent Form (last 5 matches, most recent first):"]
        for team_id, team_name in [(team1_id, team1_name), (team2_id, team2_name)]:
            matches = football_service._make_request(
                f"/teams/{team_id}/matches", params={"limit": 5, "status": "FINISHED"}
            )

            results = []
            details = []
            for match in matches.get("matches", []):
                is_home = match["homeTeam"]["id"] == team_id
                team_score = match["score"]["fullTime"]["home" if is_home else "away"]
                opp_score = match["score"]["fullTime"]["away" if is_home else "home"]
                opponent = match["awayTeam" if is_home else "homeTeam"]["shortName"]
                date = match["utcDate"].split("T")[0]

                if team_score > opp_score:
                    results.append("W")
                    details.append(f"W vs {opponent} ({team_score}-{opp_score})")
                elif team_score < opp_score:
                    results.append("L")
                    details.append(f"L vs {opponent} ({team_score}-{opp_score})")
                else:
                    results.append("D")
                    details.append(f"D vs {opponent} ({team_score}-{opp_score})")

            form = "".join(reversed(results))
            form_text.append(f"\n{team_name}:")
            form_text.append(f"Form: {form} (W=Win, D=Draw, L=Loss)")
            form_text.append("Last 5 matches:")
            for detail in reversed(details):  # Most recent first
                form_text.append(f"• {detail}")
            form_text.append("")

        await context.bot.send_message(chat_id=chat_id, text="\n".join(form_text))

    @staticmethod
    async def handle_injuries(team1_id, team2_id, teams, context, chat_id):
        """Handle injuries button"""
        team1_name = next(t["name"] for t in teams if t["id"] == team1_id)
        team2_name = next(t["name"] for t in teams if t["id"] == team2_id)

        injury_text = []
        for team_id, team_name in [(team1_id, team1_name), (team2_id, team2_name)]:
            try:
                response = football_service._make_request(f"/teams/{team_id}")
                squad = response.get("squad", [])

                # Get unavailable players (simple simulation since actual injury data might be limited)
                unavailable = [
                    player["name"]
                    for player in squad
                    if player.get("status") == "UNAVAILABLE"
                ]

                if unavailable:
                    injury_text.append(f"🏥 {team_name} unavailable players:")
                    for player in unavailable:
                        injury_text.append(f"• {player}")
                else:
                    injury_text.append(f"✅ {team_name}: No reported injuries")

            except Exception as e:
                injury_text.append(f"❌ Could not fetch injury data for {team_name}")

        await context.bot.send_message(chat_id=chat_id, text="\n\n".join(injury_text))


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if user_id not in user_states:
        user_states[user_id] = UserState()

    if query.data.startswith("league_"):
        # Handle league selection
        league_id = int(query.data.split("_")[1])
        user_selections[query.from_user.id] = {"league_id": league_id}
        user_states[user_id].league_id = league_id

        # Show teams in selected league
        teams = football_service.get_teams_in_league(league_id)
        keyboard = []
        for team in teams:
            keyboard.append(
                [
                    InlineKeyboardButton(
                        team["shortName"], callback_data=f"team_{team['id']}"
                    )
                ]
            )

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("Select a team:", reply_markup=reply_markup)

    elif query.data.startswith("team_"):
        team_id = int(query.data.split("_")[1])
        state = user_states[user_id]

        if not state.first_team:
            state.first_team = team_id
            # Show teams again for second selection
            teams = football_service.get_teams_in_league(state.league_id)
            keyboard = []
            for team in teams:
                if team["id"] != team_id:  # Don't show already selected team
                    keyboard.append(
                        [
                            InlineKeyboardButton(
                                team["shortName"], callback_data=f"team_{team['id']}"
                            )
                        ]
                    )

            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "Select second team:", reply_markup=reply_markup
            )
        else:
            state.second_team = team_id
            # Show statistics options
            keyboard = get_statistics_keyboard(state.first_team, team_id)
            await query.edit_message_text(
                "What would you like to see?", reply_markup=keyboard
            )

    elif query.data.startswith("stats_"):
        # Generate and send team comparison graph
        team1_id, team2_id = map(int, query.data.split("_")[1:])
        state = user_states[query.from_user.id]

        # Get team names
        teams = football_service.get_teams_in_league(state.league_id)
        team1_name = next(t["name"] for t in teams if t["id"] == team1_id)
        team2_name = next(t["name"] for t in teams if t["id"] == team2_id)

        # Get team stats
        team1_stats = football_service.get_team_stats(team1_name)
        team2_stats = football_service.get_team_stats(team2_name)

        # Create comparison plot
        plt.figure(figsize=(12, 8))
        plt.style.use("dark_background")

        # Categories and values for comparison
        categories = [
            "Strength",
            "Form",
            "Goals Scored",
            "Goals Conceded",
            "League Pos",
        ]
        team1_values = [
            team1_stats["strength"],
            team1_stats["form"],
            team1_stats["goals_scored_last_5"] / 15,  # Normalize to 0-1
            1 - (team1_stats["goals_conceded_last_5"] / 15),  # Inverse and normalize
            1 - (team1_stats["league_position"] / 20),  # Inverse and normalize
        ]
        team2_values = [
            team2_stats["strength"],
            team2_stats["form"],
            team2_stats["goals_scored_last_5"] / 15,
            1 - (team2_stats["goals_conceded_last_5"] / 15),
            1 - (team2_stats["league_position"] / 20),
        ]

        # Number of categories
        num_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Add the values to make a complete circle
        team1_values += team1_values[:1]
        team2_values += team2_values[:1]

        # Plot
        ax = plt.subplot(111, projection="polar")

        # Plot team 1
        ax.plot(
            angles, team1_values, "o-", linewidth=2, label=team1_name, color="#FF6B6B"
        )
        ax.fill(angles, team1_values, alpha=0.25, color="#FF6B6B")

        # Plot team 2
        ax.plot(
            angles, team2_values, "o-", linewidth=2, label=team2_name, color="#4ECDC4"
        )
        ax.fill(angles, team2_values, alpha=0.25, color="#4ECDC4")

        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.title(f"Team Comparison: {team1_name} vs {team2_name}")

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#2F3136")
        buf.seek(0)
        plt.close()

        # Send the plot
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=buf,
            caption=f"Statistical comparison between {team1_name} and {team2_name}",
        )

    elif query.data.startswith("predict_"):
        team1_id, team2_id = map(int, query.data.split("_")[1:])
        state = user_states[query.from_user.id]

        # Get team names and next match info
        teams = football_service.get_teams_in_league(state.league_id)
        team1_name = next(t["name"] for t in teams if t["id"] == team1_id)
        team2_name = next(t["name"] for t in teams if t["id"] == team2_id)

        # Get next match information
        next_match = football_service.get_next_match(team1_id, team2_id)

        # Make prediction with timeframe
        prediction = prediction_service.predict_match(
            team1_name, team2_name, timeframe=state.timeframe
        )

        # Add match time information if available
        if next_match:
            # Format match time in user-friendly way
            match_time = next_match["datetime"]
            current_time = datetime.now(match_time.tzinfo)
            days_until = (match_time.date() - current_time.date()).days

            time_str = match_time.strftime("%H:%M UTC")
            date_str = match_time.strftime("%d %B %Y")

            if days_until == 0:
                when = "Today"
            elif days_until == 1:
                when = "Tomorrow"
            else:
                when = f"In {days_until} days"

            match_info = (
                f"⚽ Next Match Information:\n"
                f"📅 Date: {when}, {date_str}\n"
                f"⏰ Time: {time_str}\n"
                f"🏟️ Venue: {next_match['venue']}\n"
                f"🏆 Competition: {next_match['competition']}\n"
                f"📊 Matchday: {next_match['matchday']}\n"
            )
            prediction = f"{match_info}\n{prediction}"
        else:
            prediction = (
                "⚠️ No scheduled match found between these teams.\n\n" + prediction
            )

        # Add odds comparison if available (with error handling)
        try:
            odds = football_service.get_match_odds(team1_id, team2_id)
            if odds:
                prediction += f"\n\n📊 Betting Odds (Win probability):"
                prediction += f"\n• {team1_name} (Home): {odds['home']:.2f} ({(1 / odds['home']) * 100:.0f}%)"
                prediction += (
                    f"\n• Draw: {odds['draw']:.2f} ({(1 / odds['draw']) * 100:.0f}%)"
                )
                prediction += f"\n• {team2_name} (Away): {odds['away']:.2f} ({(1 / odds['away']) * 100:.0f}%)"
        except Exception as e:
            logger.warning(f"Could not get odds: {e}")

        await query.edit_message_text(prediction)

        # Clear user state
        del user_states[user_id]

    elif query.data.startswith("h2h_"):
        team1_id, team2_id = map(int, query.data.split("_")[1:])
        teams = football_service.get_teams_in_league(user_states[user_id].league_id)
        await ButtonHandlers.handle_h2h(
            team1_id, team2_id, teams, context, update.effective_chat.id
        )

    elif query.data.startswith("form_"):
        team1_id, team2_id = map(int, query.data.split("_")[1:])
        teams = football_service.get_teams_in_league(user_states[user_id].league_id)
        await ButtonHandlers.handle_form(
            team1_id, team2_id, teams, context, update.effective_chat.id
        )

    elif query.data.startswith("injuries_"):
        team1_id, team2_id = map(int, query.data.split("_")[1:])
        teams = football_service.get_teams_in_league(user_states[user_id].league_id)
        await ButtonHandlers.handle_injuries(
            team1_id, team2_id, teams, context, update.effective_chat.id
        )


async def shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shutdown the bot - restricted to admins only"""
    user_id = update.effective_user.id

    if user_id not in ADMIN_USER_IDS:
        logger.warning(f"Unauthorized shutdown attempt by user {user_id}")
        await update.message.reply_text(
            "⛔ Access Denied: You are not authorized to shut down the bot."
        )
        return

    logger.info(f"Bot shutdown initiated by admin {user_id}")
    await update.message.reply_text(
        "🛑 Shutting down bot...\n"
        "Thanks for using BetWinner!\n"
        "To restart, run the bot again."
    )
    await context.application.stop()
    sys.exit(0)
