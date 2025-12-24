"""RSS feed fetcher for TBS NEWS DIG.

This module fetches and parses news articles from TBS NEWS DIG RSS feed.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

import feedparser


# Configure logging
logger = logging.getLogger(__name__)

# TBS NEWS DIG RSS feed URL
TBS_NEWS_RSS_URL = "https://newsdig.tbs.co.jp/list/feed/rss"


def fetch_news(
    rss_url: str = TBS_NEWS_RSS_URL,
    max_items: int = 20
) -> List[Dict[str, str]]:
    """Fetch and parse news articles from TBS NEWS DIG RSS feed.

    Args:
        rss_url: RSS feed URL to fetch from. Defaults to TBS NEWS DIG feed.
        max_items: Maximum number of news items to return. Defaults to 20.

    Returns:
        List of dictionaries containing news article data. Each dictionary contains:
            - title: Article title
            - link: Article URL
            - published: Publication date/time (ISO format string)
            - summary: Article summary/description

    Raises:
        Exception: If RSS feed fetch fails or parsing encounters errors.

    Examples:
        >>> articles = fetch_news()
        >>> print(f"Fetched {len(articles)} articles")
        >>> print(articles[0]['title'])
    """
    logger.info(f"Fetching RSS feed from: {rss_url}")

    try:
        # Parse RSS feed
        feed = feedparser.parse(rss_url)

        # Check for feed errors
        if feed.bozo:
            logger.warning(
                f"Feed parsing encountered issues: {feed.bozo_exception}"
            )

        # Check if feed was successfully retrieved
        if not feed.entries:
            logger.error("No entries found in RSS feed")
            raise Exception("RSS feed contains no entries")

        logger.info(f"Successfully parsed feed: {feed.feed.get('title', 'Unknown')}")
        logger.info(f"Found {len(feed.entries)} entries in feed")

        # Extract and structure article data
        articles = []
        for entry in feed.entries[:max_items]:
            article = _parse_entry(entry)
            if article:
                articles.append(article)

        logger.info(f"Successfully processed {len(articles)} articles")
        return articles

    except Exception as e:
        logger.error(f"Failed to fetch or parse RSS feed: {str(e)}")
        raise


def _parse_entry(entry: feedparser.FeedParserDict) -> Optional[Dict[str, str]]:
    """Parse a single RSS entry into structured article data.

    Args:
        entry: RSS feed entry from feedparser.

    Returns:
        Dictionary with article data, or None if required fields are missing.
    """
    try:
        # Extract publication date
        published = entry.get('published', '')
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                # Convert to ISO format for consistency
                dt = datetime(*entry.published_parsed[:6])
                published = dt.isoformat()
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse publication date: {e}")

        # Build article dictionary
        article = {
            'title': entry.get('title', '').strip(),
            'link': entry.get('link', '').strip(),
            'published': published,
            'summary': entry.get('summary', '').strip()
        }

        # Validate required fields
        if not article['title'] or not article['link']:
            logger.warning("Skipping entry with missing title or link")
            return None

        return article

    except Exception as e:
        logger.warning(f"Error parsing entry: {str(e)}")
        return None


def get_latest_articles(count: int = 5) -> List[Dict[str, str]]:
    """Fetch the latest business news articles.

    Convenience function to fetch a specific number of latest articles.

    Args:
        count: Number of latest articles to fetch. Defaults to 5.

    Returns:
        List of article dictionaries.
    """
    return fetch_news(max_items=count)


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test the fetcher
    try:
        articles = fetch_news(max_items=5)
        print(f"\nFetched {len(articles)} articles:\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Link: {article['link']}")
            print(f"   Published: {article['published']}")
            print(f"   Summary: {article['summary'][:100]}...")
            print()
    except Exception as e:
        print(f"Error: {e}")
