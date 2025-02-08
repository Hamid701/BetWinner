import requests
import logging
from config import NEWS_API_KEY, NEWS_API_BASE_URL
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NewsService:
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = NEWS_API_BASE_URL

    def get_team_news(self, team_name, limit=20):
        """Get recent news articles about a team"""
        try:
            # Calculate date range for last month
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # Make API request
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": f'"{team_name}" (football OR soccer)',
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": limit,
                    "apiKey": self.api_key,
                },
            )

            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            logger.info(f"Found {len(articles)} news articles for {team_name}")

            return [
                {
                    "title": article["title"],
                    "description": article["description"] or "",
                    "content": article["content"] or "",
                    "url": article["url"],
                    "published_at": article["publishedAt"],
                }
                for article in articles
            ]

        except Exception as e:
            logger.error(f"Error fetching news for {team_name}: {e}")
            return []

    def analyze_team_sentiment(self, articles):
        """Combine articles into a single analysis text"""
        combined_text = []

        for article in articles:
            title = article["title"]
            desc = article["description"]
            content = article["content"]

            # Combine all available text
            article_text = f"Title: {title}\n"
            if desc:
                article_text += f"Summary: {desc}\n"
            if content:
                article_text += f"Content: {content}\n"

            combined_text.append(article_text)

        return "\n---\n".join(combined_text)
