from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from .db_config import Base
import datetime


class MatchPrediction(Base):
    __tablename__ = "match_predictions"

    id = Column(Integer, primary_key=True, index=True)
    team1 = Column(String)
    team2 = Column(String)
    predicted_winner = Column(String)
    win_probability = Column(Float)
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    actual_winner = Column(String, nullable=True)
    prediction_correct = Column(Boolean, nullable=True)
    odds_home = Column(Float, nullable=True)
    odds_away = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    home_injured_players = Column(Integer, default=0)
    away_injured_players = Column(Integer, default=0)
