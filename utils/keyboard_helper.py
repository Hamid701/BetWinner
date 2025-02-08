from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def get_timeframe_keyboard():
    """Create keyboard for timeframe selection"""
    keyboard = [
        [
            InlineKeyboardButton("Last 5 matches", callback_data="timeframe_5"),
            InlineKeyboardButton("Last 10 matches", callback_data="timeframe_10"),
        ],
        [
            InlineKeyboardButton("Last 15 matches", callback_data="timeframe_15"),
            InlineKeyboardButton("Season", callback_data="timeframe_season"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_statistics_keyboard(team1_id, team2_id):
    """Create keyboard for statistics options"""
    keyboard = [
        [
            InlineKeyboardButton(
                "Head-to-Head", callback_data=f"h2h_{team1_id}_{team2_id}"
            ),
            InlineKeyboardButton(
                "Form Guide", callback_data=f"form_{team1_id}_{team2_id}"
            ),
        ],
        [
            InlineKeyboardButton(
                "Team Stats", callback_data=f"stats_{team1_id}_{team2_id}"
            ),
            InlineKeyboardButton(
                "Injuries", callback_data=f"injuries_{team1_id}_{team2_id}"
            ),
        ],
        [
            InlineKeyboardButton(
                "Make Prediction", callback_data=f"predict_{team1_id}_{team2_id}"
            ),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)
