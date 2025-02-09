import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
from database.db_config import SessionLocal
from database.models import MatchPrediction
from config import MODEL_PATH
import os
from services.football_service import FootballService
from transformers import pipeline
from services.news_service import NewsService
from services.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.scaler = self._initialize_scaler()
        self.model = self._load_model()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_service = NewsService()
        self.football_service = FootballService()  # Add this line to init

    def _initialize_scaler(self):
        """Initialize scaler with extended features"""
        scaler = StandardScaler()
        # Create sample data matching our 17 features
        sample_features = np.array(
            [
                # [str, form, h2h, goals_f, goals_a, home_adv, inj, league_pos] x2 teams + sentiment
                [0.75, 0.8, 2, 8, 3, 0.2, 0.1, 3, 0.7, 0.6, 1, 6, 5, 0.1, 0.2, 5, 0.5],
                [0.65, 0.7, 1, 5, 4, 0.1, 0.3, 8, 0.8, 0.9, 3, 9, 2, 0.2, 0.1, 2, -0.2],
                [0.80, 0.9, 3, 10, 2, 0.3, 0.0, 1, 0.7, 0.7, 2, 7, 6, 0.1, 0.2, 6, 0.0],
            ]
        )
        scaler.fit(sample_features)
        return scaler

    def _create_initial_training_data(self):
        """Create initial training data with extended features"""
        X = np.array(
            [
                # [str, form, h2h, goals_f, goals_a, home_adv, inj, league_pos] x2 teams + sentiment
                [
                    0.75,
                    0.8,
                    2,
                    8,
                    3,
                    0.2,
                    0.1,
                    3,
                    0.7,
                    0.6,
                    1,
                    6,
                    5,
                    0.1,
                    0.2,
                    5,
                    0.5,
                ],  # Team1 wins
                [
                    0.65,
                    0.7,
                    1,
                    5,
                    4,
                    0.1,
                    0.3,
                    8,
                    0.8,
                    0.9,
                    3,
                    9,
                    2,
                    0.2,
                    0.1,
                    2,
                    -0.2,
                ],  # Team2 wins
                [
                    0.80,
                    0.9,
                    3,
                    10,
                    2,
                    0.3,
                    0.0,
                    1,
                    0.7,
                    0.7,
                    2,
                    7,
                    6,
                    0.1,
                    0.2,
                    6,
                    0.0,
                ],  # Team1 wins
            ]
        )
        y = np.array([1, 0, 1])  # 1 for team1 wins, 0 for team2 wins
        return X, y

    def _load_model(self):
        """Load or create the prediction model"""
        try:
            if os.path.exists(MODEL_PATH):
                logger.info("Loading existing model...")
                # Add version warning filter
                import warnings

                warnings.filterwarnings("ignore", category=UserWarning)
                model = joblib.load(MODEL_PATH)
                # Verify model with scaled dummy data
                dummy_data = self.scaler.transform([[0] * 17])
                model.predict(dummy_data)
                logger.info("Existing model loaded successfully")
                return model
            else:
                logger.info("No existing model found, creating new one...")
                return self._create_new_model()
        except Exception as e:
            logger.warning(f"Error loading model: {e}, creating new one...")
            return self._create_new_model()

    def _create_new_model(self):
        """Create and train a new model with initial data"""
        try:
            logger.info("Creating new logistic regression model...")
            model = LogisticRegression(
                random_state=42, class_weight="balanced", max_iter=1000, solver="lbfgs"
            )

            # Get and scale initial training data
            X, y = self._create_initial_training_data()
            X_scaled = self.scaler.transform(X)

            # Fit model with scaled data
            model.fit(X_scaled, y)

            # Save the model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            logger.info("New model created and saved successfully")

            return model

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

    def train_model(self, features, labels):
        try:
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, labels)
            joblib.dump(self.model, MODEL_PATH)
            logger.info("Model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

    def analyze_sentiment(self, team1, team2):
        try:
            # Get news articles for both teams
            team1_news = self.news_service.get_team_news(team1, limit=20)
            team2_news = self.news_service.get_team_news(team2, limit=20)

            # Prepare texts for analysis
            team1_texts = [
                self.news_service.analyze_team_sentiment([article])
                for article in team1_news
            ]
            team2_texts = [
                self.news_service.analyze_team_sentiment([article])
                for article in team2_news
            ]

            # Analyze sentiment for each team
            team1_sentiment = self.sentiment_analyzer.analyze_batch(team1_texts)
            team2_sentiment = self.sentiment_analyzer.analyze_batch(team2_texts)

            # Calculate relative sentiment (difference between teams)
            relative_sentiment = (team1_sentiment - team2_sentiment) / 2

            logger.info(
                f"Team sentiments - {team1}: {team1_sentiment:.2f}, "
                f"{team2}: {team2_sentiment:.2f}, "
                f"Relative: {relative_sentiment:.2f}"
            )

            return relative_sentiment, False

        except Exception as e:
            logger.warning(f"Using fallback sentiment analysis due to error: {e}")
            return np.random.normal(0.0, 0.3), True

    def _create_match_analysis_text(self, team1, team2):
        """Create analysis text for the match"""
        football_service = FootballService()
        team1_stats = football_service.get_team_stats(team1)
        team2_stats = football_service.get_team_stats(team2)

        # Create more detailed analysis text
        analysis_text = (
            f"Pre-match analysis: {team1} vs {team2}. "
            f"{team1} is currently in excellent form ({team1_stats['form']:.2f}/1.0) "
            f"and ranked #{team1_stats['league_position']} in the league with a strength rating of {team1_stats['strength']:.2f}. "
            f"Their recent performance shows {team1_stats['goals_scored_last_5']} goals scored and {team1_stats['goals_conceded_last_5']} conceded. "
            f"Meanwhile, {team2} has shown {team2_stats['form']:.2f}/1.0 form "
            f"and sits at #{team2_stats['league_position']} with {team2_stats['goals_scored_last_5']} goals scored and {team2_stats['goals_conceded_last_5']} conceded. "
            f"Historical head-to-head shows {team1} with {team1_stats['head_to_head_wins']} wins versus {team2}'s {team2_stats['head_to_head_wins']}. "
            f"This promises to be an exciting match between two competitive teams."
        )

        # Add news summary
        team1_news = self.news_service.get_team_news(team1)
        team2_news = self.news_service.get_team_news(team2)

        news_summary = (
            f"Recent news coverage shows "
            f"{len(team1_news)} articles about {team1} and "
            f"{len(team2_news)} articles about {team2} "
            f"in the past week."
        )

        analysis_text = f"{analysis_text} {news_summary}"

        return analysis_text

    def predict_match(self, team1, team2, timeframe=5):
        try:
            # Get stats with timeframe
            team1_stats = self.football_service.get_team_stats(
                team1, timeframe=timeframe
            )
            team2_stats = self.football_service.get_team_stats(
                team2, timeframe=timeframe
            )

            # Add warning for unknown teams
            prediction_warnings = []
            if team1_stats == self.football_service._get_default_stats():
                prediction_warnings.append(f"⚠️ Limited data available for {team1}")
            if team2_stats == self.football_service._get_default_stats():
                prediction_warnings.append(f"⚠️ Limited data available for {team2}")

            # Update this line to get both sentiment and fallback flag
            sentiment_score, is_fallback = self.analyze_sentiment(team1, team2)
            logger.info(f"Sentiment score: {sentiment_score} (fallback: {is_fallback})")

            # Prepare feature vector [17 features total]
            features = [
                # Team 1: 8 features
                team1_stats.get("strength", 0),  # Overall strength (0-1)
                team1_stats.get("form", 0),  # Recent form (0-1)
                team1_stats.get("head_to_head_wins", 0),  # H2H wins count
                team1_stats.get("goals_scored_last_5", 0),
                team1_stats.get("goals_conceded_last_5", 0),
                team1_stats.get("home_advantage", 0),  # Home/Away impact (-1 to 1)
                team1_stats.get("injuries_impact", 0),  # Impact of injuries (0-1)
                team1_stats.get("league_position", 10),  # Current position (1-20)
                # Team 2: 8 features (same as above)
                team2_stats.get("strength", 0),
                team2_stats.get("form", 0),
                team2_stats.get("head_to_head_wins", 0),
                team2_stats.get("goals_scored_last_5", 0),
                team2_stats.get("goals_conceded_last_5", 0),
                team2_stats.get("home_advantage", 0),
                team2_stats.get("injuries_impact", 0),
                team2_stats.get("league_position", 10),
                # Context: 1 feature
                sentiment_score,  # News sentiment (-1 to 1)
            ]

            # Scale features using StandardScaler
            features_scaled = self.scaler.transform([features])

            # Get prediction probabilities using LogisticRegression
            prediction_prob = self.model.predict_proba(features_scaled)[0]

            # Determine winner and confidence
            team1_win_prob = prediction_prob[1]  # Probability of team1 winning
            winner = team1 if team1_win_prob > 0.5 else team2
            confidence = max(prediction_prob)

            # Add confidence adjustment based on data quality
            if team1_stats == self.football_service._get_default_stats():
                confidence *= 0.7  # Reduce confidence for unknown teams
            if team2_stats == self.football_service._get_default_stats():
                confidence *= 0.7

            # Add league level check
            if team1_stats.get("league_level", 1) != team2_stats.get("league_level", 1):
                confidence *= 0.8  # Reduce confidence for cross-league matches

            # Adjust confidence based on data quality and context
            # Reduce extremely high confidence predictions
            if confidence > 0.85:
                confidence = (
                    0.85 + (confidence - 0.85) * 0.5
                )  # Dampen very high confidence

            # Adjust for home advantage
            home_advantage = team1_stats.get("home_advantage", 0)
            if home_advantage > 0:
                team1_win_prob = min(team1_win_prob * (1 + home_advantage * 0.2), 0.95)

            # Factor in recent form more heavily
            form_diff = team1_stats.get("form", 0) - team2_stats.get("form", 0)
            confidence = confidence * (1 + form_diff * 0.1)

            # Cap final confidence
            confidence = min(max(confidence, 0.55), 0.95)

            # Get additional stats
            team1_additional = {
                "goals_scored": team1_stats["goals_scored_last_5"],
                "goals_conceded": team1_stats["goals_conceded_last_5"],
                "h2h_wins": team1_stats["head_to_head_wins"],
                "home_adv": team1_stats["home_advantage"],
                "league_pos": team1_stats["league_position"],
            }

            team2_additional = {
                "goals_scored": team2_stats["goals_scored_last_5"],
                "goals_conceded": team2_stats["goals_conceded_last_5"],
                "h2h_wins": team2_stats["head_to_head_wins"],
                "home_adv": team2_stats["home_advantage"],
                "league_pos": team2_stats["league_position"],
            }

            # Enhanced output format with more details
            prediction_details = [
                f"🏆 Match Prediction: {team1} vs {team2}",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"Predicted Winner: {winner} 🎯",
                f"Confidence: {confidence:.1%}",
                f"",
                f"Team Analysis:",
                f"• {team1}:",
                f"  - Overall Rating: {team1_stats['strength']:.2f}",
                f"  - Recent Form: {team1_stats['form']:.2f}",
                f"  - League Position: #{team1_additional['league_pos']}",
                f"  - Recent Performance: {team1_additional['goals_scored']} goals for, {team1_additional['goals_conceded']} against",
                f"  - Head-to-Head Wins: {team1_additional['h2h_wins']}",
                f"",
                f"• {team2}:",
                f"  - Overall Rating: {team2_stats['strength']:.2f}",
                f"  - Recent Form: {team2_stats['form']:.2f}",
                f"  - League Position: #{team2_additional['league_pos']}",
                f"  - Recent Performance: {team2_additional['goals_scored']} goals for, {team2_additional['goals_conceded']} against",
                f"  - Head-to-Head Wins: {team2_additional['h2h_wins']}",
                f"",
                f"Match Context:",
                f"- Sentiment Analysis: {sentiment_score:+.2f}{' (based on current form and history)' if not is_fallback else ' (fallback)'}",
                f"- Venue Impact: {team1_additional['home_adv']:.2f} vs {team2_additional['home_adv']:.2f}",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"Based on current form, historical data, and team statistics",
            ]

            # Add warnings to prediction details if any
            if prediction_warnings:
                prediction_details.insert(1, "\n".join(prediction_warnings))

            # Add prediction tracking
            self._save_prediction(
                team1=team1,
                team2=team2,
                winner=winner,
                probability=confidence,
                sentiment=sentiment_score,
                odds_home=team1_stats.get("odds", {}).get("home"),
                odds_away=team2_stats.get("odds", {}).get("away"),
                odds_draw=team1_stats.get("odds", {}).get("draw"),
                home_injuries=len(team1_stats.get("injuries", [])),
                away_injuries=len(team2_stats.get("injuries", [])),
            )

            return "\n".join(prediction_details)

        except Exception as e:
            logger.error(f"Error in match prediction: {str(e)}")
            return f"⚠️ Error making prediction: {str(e)}"

    def _prepare_features(self, team1_stats, team2_stats, sentiment):
        """Prepare extended feature set"""
        return [
            # Team 1 features
            team1_stats.get("strength", 0),
            team1_stats.get("form", 0),
            team1_stats.get("head_to_head_wins", 0),
            team1_stats.get("goals_scored_last_5", 0),
            team1_stats.get("goals_conceded_last_5", 0),
            team1_stats.get("home_advantage", 0),
            team1_stats.get("injuries_impact", 0),
            team1_stats.get("league_position", 10),
            # Team 2 features
            team2_stats.get("strength", 0),
            team2_stats.get("form", 0),
            team2_stats.get("head_to_head_wins", 0),
            team2_stats.get("goals_scored_last_5", 0),
            team2_stats.get("goals_conceded_last_5", 0),
            team2_stats.get("home_advantage", 0),
            team2_stats.get("injuries_impact", 0),
            team2_stats.get("league_position", 10),
            # Match context
            sentiment,
        ]

    def _get_feature_importance(self, scaled_features):
        """Get feature importance with extended features"""
        feature_names = [
            "Team 1 Strength",
            "Team 1 Form",
            "Team 1 H2H Wins",
            "Team 1 Goals Scored",
            "Team 1 Goals Conceded",
            "Team 1 Home Advantage",
            "Team 1 Injuries Impact",
            "Team 1 League Position",
            "Team 2 Strength",
            "Team 2 Form",
            "Team 2 H2H Wins",
            "Team 2 Goals Scored",
            "Team 2 Goals Conceded",
            "Team 2 Home Advantage",
            "Team 2 Injuries Impact",
            "Team 2 League Position",
            "Sentiment",
        ]
        coefficients = self.model.coef_[0]
        importance = coefficients * scaled_features
        return dict(
            sorted(
                zip(feature_names, abs(importance)),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )

    def _save_prediction(
        self,
        team1,
        team2,
        winner,
        probability,
        sentiment,
        odds_home=None,
        odds_away=None,
        odds_draw=None,
        home_injuries=None,
        away_injuries=None,
    ):
        db = SessionLocal()
        try:
            prediction = MatchPrediction(
                team1=team1,
                team2=team2,
                predicted_winner=winner,
                win_probability=probability,
                sentiment_score=sentiment,
                odds_home=odds_home,
                odds_away=odds_away,
                odds_draw=odds_draw,
                home_injured_players=home_injuries,  # Fixed column name
                away_injured_players=away_injuries,  # Fixed column name
            )
            db.add(prediction)
            db.commit()
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
        finally:
            db.close()
