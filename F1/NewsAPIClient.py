import warnings
warnings.filterwarnings('ignore', message='`resume_download` is deprecated')

import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from FakeNewsDetector import FakeNewsDetector


# Class for fetching news articles from NewsAPI.
class NewsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/top-headlines"
    
    # Fetching news articles from NewsAPI.
    def fetch_news(self, country: str = "us", category: str = "general", page_size: int = 100) -> List[Dict]:
        params = {
            "apiKey": self.api_key,
            "country": country,
            "category": category,
            "pageSize": page_size,
        }
        try:
            print(f"\nFetching news from NewsAPI (category: {category})...")
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if response.status_code != 200:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                print(f"Status code: {response.status_code}")
                return []

            if data.get("status") != "ok":
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return []

            articles = data.get("articles", [])
            print(f"Successfully fetched {len(articles)} articles")

            return articles

        except requests.exceptions.RequestException as e:
            print(f"Network error while fetching news: {e}")
            return []

    # Removing source names from the text.
    @staticmethod
    def remove_source_names(text: str) -> str:
        return FakeNewsDetector.remove_source_names(text)

    # Extracting and combining title and description from an article.
    @staticmethod
    def extract_title_and_description(article: Dict) -> Optional[str]:
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()

        if not title:
            return None

        title = FakeNewsDetector.remove_source_names(title)
        description = FakeNewsDetector.remove_source_names(description)
        
        return f"{title} - {description}" if description else title

    # Filtering articles by publication time.
    @staticmethod
    def filter_by_time(articles: List[Dict], hours: int) -> List[Dict]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        filtered = []
        
        print(f"\nArticles by (last {hours} hours)...")
        for article in articles:
            published_at = article.get("publishedAt")
            if not published_at:
                continue
                
            try:
                pub_time = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if pub_time > now:
                    pub_time = now
                
                if pub_time >= cutoff:
                    filtered.append(article)
            except Exception as e:
                print(f"Warning: Could not parse date {published_at}: {e}")
                filtered.append(article)
                
        return filtered

    # Getting news articles filtered by timeframe.
    def get_news_by_timeframe(self, timeframe_hours: int, country: str = "us", category: str = "general") -> List[str]:
        articles = self.fetch_news(country, category)
        if not articles:
            return []
        filtered = self.filter_by_time(articles, timeframe_hours)
        processed = [text for text in (self.extract_title_and_description(a) for a in filtered) if text]
        return processed