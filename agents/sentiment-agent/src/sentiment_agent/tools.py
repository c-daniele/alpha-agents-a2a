"""Tools for the Sentiment Agent - News collection and sentiment analysis."""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import dateutil.parser
import feedparser
import httpx
import yfinance as yf
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class NewsCollectionInput(BaseModel):
    """Input schema for news collection."""

    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    max_articles: int = Field(
        default=10,
        description="Maximum number of news articles to collect"
    )
    lookback_days: int = Field(
        default=7,
        description="Number of days to look back for news"
    )


class SentimentAnalysisInput(BaseModel):
    """Input schema for sentiment analysis."""

    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    news_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-collected news data (if not provided, will collect news first)"
    )


class CompanyResolverInput(BaseModel):
    """Input schema for company name resolution."""

    query: str = Field(description="Company name, ISIN, or potential ticker symbol to resolve")


class StockNewsCollectionTool(BaseTool):
    """Tool to collect stock-related news from multiple sources."""

    name: str = "collect_stock_news"
    description: str = (
        "Collects recent financial news related to a specific stock from multiple sources "
        "including Yahoo Finance, Google News, and financial RSS feeds. Returns structured "
        "news data with headlines, sources, publication dates, and content."
    )
    args_schema: type[BaseModel] = NewsCollectionInput

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        user_agent = "Mozilla/5.0 (compatible; SentimentAgent/1.0)"
        return httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": user_agent}
        )

    def _run(self, symbol: str, max_articles: int = 10, lookback_days: int = 7) -> Dict[str, Any]:
        """Collect stock news synchronously."""
        return asyncio.run(self._arun(symbol, max_articles, lookback_days))

    async def _arun(self, symbol: str, max_articles: int = 10, lookback_days: int = 7) -> Dict[str, Any]:
        """Collect stock news asynchronously."""
        session = None
        try:
            # Get company information first
            ticker = yf.Ticker(symbol.upper())
            try:
                info = ticker.info
                company_name = info.get('longName', symbol.upper())
            except:
                company_name = symbol.upper()

            logger.info(f"Collecting news for {symbol} ({company_name})")

            # Create session for news collection
            session = await self._get_session()

            # Collect news from multiple sources
            all_articles = []

            # Source 1: Yahoo Finance RSS
            yahoo_articles = await self._collect_yahoo_finance_news(session, symbol, max_articles // 2)
            all_articles.extend(yahoo_articles)

            # Source 2: Google News
            google_articles = await self._collect_google_news(session, symbol, company_name, max_articles // 2)
            all_articles.extend(google_articles)

            # Filter by date and deduplicate
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            filtered_articles = []
            seen_titles = set()

            for article in all_articles:
                # Skip duplicates based on title similarity
                title_key = self._normalize_title(article.get('title', ''))
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                # Check date
                pub_date = article.get('published_date')
                if pub_date:
                    # Ensure both dates are timezone-aware for comparison
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    if pub_date >= cutoff_date:
                        filtered_articles.append(article)
                else:
                    # Include articles without dates
                    filtered_articles.append(article)

            # Sort by date (newest first) and limit
            filtered_articles.sort(key=lambda x: x.get('published_date', datetime.min), reverse=True)
            final_articles = filtered_articles[:max_articles]

            result = {
                "success": True,
                "symbol": symbol.upper(),
                "company_name": company_name,
                "collection_date": datetime.now(timezone.utc).isoformat(),
                "lookback_days": lookback_days,
                "articles_collected": len(final_articles),
                "articles": final_articles
            }

            logger.info(f"Successfully collected {len(final_articles)} articles for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Error collecting news for {symbol}: {str(e)}")
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Failed to collect news: {str(e)}"
            }

    async def _collect_yahoo_finance_news(self, session: httpx.AsyncClient, symbol: str, max_articles: int) -> List[Dict[str, Any]]:
        """Collect news from Yahoo Finance."""
        articles = []
        try:
            # Yahoo Finance RSS feed
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

            response = await session.get(rss_url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)

            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get('title', ''),
                    "url": entry.get('link', ''),
                    "source": "Yahoo Finance",
                    "published_date": self._parse_date(entry.get('published', '')),
                    "summary": entry.get('summary', ''),
                    "content": entry.get('summary', '')  # RSS usually only has summary
                }
                articles.append(article)

        except Exception as e:
            logger.warning(f"Failed to collect Yahoo Finance news: {str(e)}")

        return articles

    async def _collect_google_news(self, session: httpx.AsyncClient, symbol: str, company_name: str, max_articles: int) -> List[Dict[str, Any]]:
        """Collect news from Google News RSS."""
        articles = []
        try:
            # Google News RSS search
            query = f'"{company_name}" OR "{symbol}" stock finance'
            encoded_query = quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            response = await session.get(rss_url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)

            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get('title', ''),
                    "url": entry.get('link', ''),
                    "source": self._extract_source_from_google_news(entry),
                    "published_date": self._parse_date(entry.get('published', '')),
                    "summary": entry.get('title', ''),  # Google News RSS doesn't have summary
                    "content": entry.get('title', '')
                }
                articles.append(article)

        except Exception as e:
            logger.warning(f"Failed to collect Google News: {str(e)}")

        return articles

    def _extract_source_from_google_news(self, entry) -> str:
        """Extract source from Google News entry."""
        try:
            # Google News includes source in the title usually
            title = entry.get('title', '')
            if ' - ' in title:
                return title.split(' - ')[-1]
            return "Google News"
        except:
            return "Google News"

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None

        try:
            # Try parsing common date formats
            parsed_date = dateutil.parser.parse(date_str)
            # Ensure timezone-aware datetime
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            return parsed_date
        except:
            return None

    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication."""
        # Remove special characters, convert to lowercase, remove extra spaces
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized[:50]  # Take first 50 chars for comparison


class CompanyNameResolverTool(BaseTool):
    """Tool to resolve company names or ISIN to stock tickers."""

    name: str = "resolve_company_ticker"
    description: str = (
        "Resolves company names (e.g., 'Apple', 'Microsoft') or ISIN codes to stock tickers. "
        "Returns the ticker symbol that can be used for news collection."
    )
    args_schema: type[BaseModel] = CompanyResolverInput

    def _run(self, query: str) -> Dict[str, Any]:
        """Resolve company name/ISIN to ticker symbol."""
        try:
            query = query.strip()

            # Common company name mappings for financial news
            company_mappings = {
                # Tech companies
                "apple": "AAPL",
                "apple inc": "AAPL",
                "microsoft": "MSFT",
                "microsoft corp": "MSFT",
                "google": "GOOGL",
                "alphabet": "GOOGL",
                "amazon": "AMZN",
                "amazon.com": "AMZN",
                "tesla": "TSLA",
                "tesla inc": "TSLA",
                "meta": "META",
                "facebook": "META",
                "netflix": "NFLX",
                "nvidia": "NVDA",
                "nvidia corp": "NVDA",

                # Financial companies
                "jpmorgan": "JPM",
                "jp morgan": "JPM",
                "bank of america": "BAC",
                "goldman sachs": "GS",
                "morgan stanley": "MS",
                "wells fargo": "WFC",

                # Other major companies
                "walmart": "WMT",
                "berkshire hathaway": "BRK-B",
                "johnson & johnson": "JNJ",
                "exxon mobil": "XOM",
                "unitedhealth": "UNH",
                "procter & gamble": "PG",
                "visa": "V",
                "home depot": "HD",
                "mastercard": "MA",
                "coca cola": "KO",
                "coca-cola": "KO",
                "disney": "DIS",
                "walt disney": "DIS",
            }

            # Check if it's already a ticker
            if query.isupper() and 1 <= len(query) <= 5 and query.isalpha():
                try:
                    ticker = yf.Ticker(query)
                    info = ticker.info
                    if info and 'symbol' in info:
                        return {
                            "success": True,
                            "query": query,
                            "ticker": query,
                            "company_name": info.get('longName', 'Unknown'),
                            "resolution_method": "direct_ticker"
                        }
                except:
                    pass

            # Check company mappings
            query_lower = query.lower()
            if query_lower in company_mappings:
                ticker = company_mappings[query_lower]
                try:
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    return {
                        "success": True,
                        "query": query,
                        "ticker": ticker,
                        "company_name": info.get('longName', 'Unknown'),
                        "resolution_method": "company_mapping"
                    }
                except Exception as e:
                    logger.warning(f"Validation failed for mapped ticker {ticker}: {e}")

            return {
                "success": False,
                "query": query,
                "error": f"Could not resolve '{query}' to a valid stock ticker",
                "suggestions": [
                    "Try using the stock ticker directly (e.g., AAPL for Apple)",
                    "Check spelling of company name",
                    "Use well-known company names like 'Apple', 'Microsoft', 'Google', etc."
                ]
            }

        except Exception as e:
            logger.error(f"Error resolving ticker for {query}: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": f"Failed to resolve ticker: {str(e)}"
            }

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Resolve company name/ISIN to ticker symbol asynchronously."""
        return await asyncio.to_thread(self._run, query)


# Export all tools
def get_sentiment_tools() -> List[BaseTool]:
    """Get all sentiment analysis tools."""
    return [
        CompanyNameResolverTool(),
        StockNewsCollectionTool()
    ]
