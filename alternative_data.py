import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp
from dataclasses import dataclass
import time
from textblob import TextBlob
import re
import feedparser
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE
import sqlite3

@dataclass
class NewsEvent:
    """Container for news events"""
    timestamp: datetime
    title: str
    content: str
    source: str
    symbols_mentioned: List[str]
    sentiment_score: float
    impact_score: float
    category: str
    url: str = ""

@dataclass
class EconomicEvent:
    """Container for economic events"""
    timestamp: datetime
    country: str
    event_name: str
    importance: str  # High, Medium, Low
    actual_value: Optional[float]
    forecast_value: Optional[float]
    previous_value: Optional[float]
    currency: str
    impact_score: float

@dataclass
class SocialSentiment:
    """Container for social media sentiment"""
    timestamp: datetime
    platform: str
    symbol: str
    sentiment_score: float
    volume_score: float
    mentions_count: int
    bullish_ratio: float
    bearish_ratio: float

class NewsDataProvider:
    """Fetches and processes financial news data"""
    
    def __init__(self):
        self.logger = logging.getLogger('NewsDataProvider')
        self.news_sources = {
            'reuters': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'investing': 'https://www.investing.com/rss/news.rss'
        }
        
    async def fetch_financial_news(self, symbols: List[str] = None, 
                                 hours_back: int = 24) -> List[NewsEvent]:
        """Fetch recent financial news"""
        try:
            all_news = []
            
            for source_name, feed_url in self.news_sources.items():
                try:
                    news_items = await self._fetch_rss_feed(source_name, feed_url)
                    filtered_news = self._filter_relevant_news(news_items, symbols)
                    all_news.extend(filtered_news)
                except Exception as e:
                    self.logger.warning(f"Error fetching from {source_name}: {e}")
                    continue
            
            # Sort by timestamp and filter by time
            cutoff_time = datetime.now(TIMEZONE) - timedelta(hours=hours_back)
            recent_news = [
                news for news in all_news 
                if news.timestamp >= cutoff_time
            ]
            
            # Remove duplicates based on title similarity
            unique_news = self._remove_duplicate_news(recent_news)
            
            self.logger.info(f"Fetched {len(unique_news)} unique news items")
            return unique_news
            
        except Exception as e:
            self.logger.error(f"Error fetching financial news: {e}")
            return []
    
    async def _fetch_rss_feed(self, source_name: str, feed_url: str) -> List[NewsEvent]:
        """Fetch and parse RSS feed"""
        try:
            # Use feedparser to get RSS data
            feed = feedparser.parse(feed_url)
            news_items = []
            
            for entry in feed.entries:
                try:
                    # Parse timestamp
                    if hasattr(entry, 'published_parsed'):
                        timestamp = datetime(*entry.published_parsed[:6])
                        timestamp = timestamp.replace(tzinfo=TIMEZONE)
                    else:
                        timestamp = datetime.now(TIMEZONE)
                    
                    # Extract content
                    title = entry.get('title', '')
                    content = entry.get('summary', entry.get('description', ''))
                    url = entry.get('link', '')
                    
                    # Clean content
                    content = self._clean_text(content)
                    
                    # Calculate sentiment
                    sentiment_score = self._calculate_sentiment(title + " " + content)
                    
                    # Determine symbols mentioned
                    symbols_mentioned = self._extract_symbols(title + " " + content)
                    
                    # Calculate impact score
                    impact_score = self._calculate_impact_score(title, content, source_name)
                    
                    news_event = NewsEvent(
                        timestamp=timestamp,
                        title=title,
                        content=content,
                        source=source_name,
                        symbols_mentioned=symbols_mentioned,
                        sentiment_score=sentiment_score,
                        impact_score=impact_score,
                        category=self._categorize_news(title + " " + content),
                        url=url
                    )
                    
                    news_items.append(news_event)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing news entry: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score using TextBlob"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            return float(sentiment)  # Range: -1 to 1
        except Exception:
            return 0.0
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract currency symbols mentioned in text"""
        symbols = []
        text_upper = text.upper()
        
        # Common currency pairs and symbols
        currency_patterns = [
            'EUR/USD', 'EURUSD', 'GBP/USD', 'GBPUSD', 'USD/JPY', 'USDJPY',
            'AUD/USD', 'AUDUSD', 'USD/CAD', 'USDCAD', 'USD/CHF', 'USDCHF',
            'EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD',
            'GOLD', 'XAU', 'SILVER', 'XAG', 'OIL', 'CRUDE'
        ]
        
        for pattern in currency_patterns:
            if pattern in text_upper:
                symbols.append(pattern)
        
        return list(set(symbols))  # Remove duplicates
    
    def _calculate_impact_score(self, title: str, content: str, source: str) -> float:
        """Calculate potential market impact score"""
        score = 0.0
        text = (title + " " + content).lower()
        
        # High impact keywords
        high_impact_words = [
            'central bank', 'fed', 'ecb', 'boe', 'boj', 'interest rate',
            'monetary policy', 'inflation', 'gdp', 'unemployment',
            'trade war', 'brexit', 'crisis', 'recession', 'stimulus'
        ]
        
        medium_impact_words = [
            'earnings', 'forecast', 'outlook', 'guidance', 'revenue',
            'profit', 'loss', 'merger', 'acquisition', 'ipo'
        ]
        
        # Calculate base score
        for word in high_impact_words:
            if word in text:
                score += 0.3
        
        for word in medium_impact_words:
            if word in text:
                score += 0.1
        
        # Source credibility multiplier
        source_multipliers = {
            'reuters': 1.2,
            'bloomberg': 1.2,
            'marketwatch': 1.0,
            'investing': 0.8
        }
        
        score *= source_multipliers.get(source, 1.0)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news by topic"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['central bank', 'fed', 'ecb', 'interest rate', 'monetary']):
            return 'monetary_policy'
        elif any(word in text_lower for word in ['gdp', 'inflation', 'unemployment', 'economic']):
            return 'economic_data'
        elif any(word in text_lower for word in ['trade', 'tariff', 'brexit', 'political']):
            return 'geopolitical'
        elif any(word in text_lower for word in ['earnings', 'revenue', 'profit', 'corporate']):
            return 'corporate'
        else:
            return 'general'
    
    def _filter_relevant_news(self, news_items: List[NewsEvent], 
                            symbols: List[str] = None) -> List[NewsEvent]:
        """Filter news relevant to specified symbols"""
        if not symbols:
            return news_items
        
        relevant_news = []
        symbols_upper = [s.upper() for s in symbols]
        
        for news in news_items:
            # Check if any target symbols are mentioned
            news_symbols = [s.upper() for s in news.symbols_mentioned]
            if any(symbol in news_symbols for symbol in symbols_upper):
                relevant_news.append(news)
            # Also include high-impact general news
            elif news.impact_score > 0.5:
                relevant_news.append(news)
        
        return relevant_news
    
    def _remove_duplicate_news(self, news_items: List[NewsEvent]) -> List[NewsEvent]:
        """Remove duplicate news based on title similarity"""
        unique_news = []
        seen_titles = set()
        
        for news in sorted(news_items, key=lambda x: x.timestamp, reverse=True):
            # Simple duplicate detection based on first 50 characters of title
            title_key = news.title[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)
        
        return unique_news

class EconomicCalendarProvider:
    """Fetches economic calendar data"""
    
    def __init__(self):
        self.logger = logging.getLogger('EconomicCalendarProvider')
        
    async def fetch_economic_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Fetch upcoming economic events"""
        try:
            # This would typically connect to a paid service like Trading Economics API
            # For demo purposes, we'll create sample events
            
            events = []
            base_time = datetime.now(TIMEZONE)
            
            # Sample economic events (in production, fetch from real API)
            sample_events = [
                {
                    'name': 'Non-Farm Payrolls',
                    'country': 'US',
                    'currency': 'USD',
                    'importance': 'High',
                    'days_offset': 3,
                    'hour': 13,
                    'minute': 30
                },
                {
                    'name': 'ECB Interest Rate Decision',
                    'country': 'EU',
                    'currency': 'EUR',
                    'importance': 'High',
                    'days_offset': 1,
                    'hour': 13,
                    'minute': 45
                },
                {
                    'name': 'GDP Growth Rate',
                    'country': 'UK',
                    'currency': 'GBP',
                    'importance': 'Medium',
                    'days_offset': 2,
                    'hour': 9,
                    'minute': 30
                },
                {
                    'name': 'Inflation Rate',
                    'country': 'US',
                    'currency': 'USD',
                    'importance': 'High',
                    'days_offset': 5,
                    'hour': 13,
                    'minute': 30
                }
            ]
            
            for event_data in sample_events:
                if event_data['days_offset'] <= days_ahead:
                    event_time = base_time + timedelta(
                        days=event_data['days_offset'],
                        hours=event_data['hour'] - base_time.hour,
                        minutes=event_data['minute'] - base_time.minute
                    )
                    
                    # Generate realistic forecast/previous values
                    base_value = np.random.normal(2.0, 0.5)
                    forecast = round(base_value + np.random.normal(0, 0.1), 1)
                    previous = round(base_value + np.random.normal(0, 0.2), 1)
                    
                    event = EconomicEvent(
                        timestamp=event_time,
                        country=event_data['country'],
                        event_name=event_data['name'],
                        importance=event_data['importance'],
                        actual_value=None,  # Not yet released
                        forecast_value=forecast,
                        previous_value=previous,
                        currency=event_data['currency'],
                        impact_score=self._calculate_event_impact(event_data)
                    )
                    
                    events.append(event)
            
            self.logger.info(f"Fetched {len(events)} economic events")
            return events
            
        except Exception as e:
            self.logger.error(f"Error fetching economic events: {e}")
            return []
    
    def _calculate_event_impact(self, event_data: Dict) -> float:
        """Calculate expected market impact of economic event"""
        importance_scores = {
            'High': 0.8,
            'Medium': 0.5,
            'Low': 0.2
        }
        
        base_score = importance_scores.get(event_data['importance'], 0.2)
        
        # High-impact events
        high_impact_events = [
            'Non-Farm Payrolls', 'Interest Rate Decision', 'FOMC', 'ECB',
            'GDP', 'Inflation', 'Unemployment'
        ]
        
        if any(keyword in event_data['name'] for keyword in high_impact_events):
            base_score *= 1.2
        
        return min(base_score, 1.0)

class SocialSentimentProvider:
    """Analyzes social media sentiment (simulated)"""
    
    def __init__(self):
        self.logger = logging.getLogger('SocialSentimentProvider')
        
    async def fetch_social_sentiment(self, symbols: List[str]) -> List[SocialSentiment]:
        """Fetch social media sentiment for symbols"""
        try:
            sentiments = []
            
            # In production, this would connect to Twitter API, Reddit API, etc.
            # For demo, we'll simulate sentiment data
            
            for symbol in symbols:
                # Simulate sentiment data
                sentiment_data = self._generate_simulated_sentiment(symbol)
                sentiments.append(sentiment_data)
            
            self.logger.info(f"Fetched sentiment for {len(symbols)} symbols")
            return sentiments
            
        except Exception as e:
            self.logger.error(f"Error fetching social sentiment: {e}")
            return []
    
    def _generate_simulated_sentiment(self, symbol: str) -> SocialSentiment:
        """Generate realistic sentiment data for simulation"""
        
        # Base sentiment around neutral with some volatility
        base_sentiment = np.random.normal(0, 0.3)
        base_sentiment = max(-1, min(1, base_sentiment))  # Clamp to [-1, 1]
        
        # Volume based on symbol popularity
        popular_symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'XAU/USD']
        base_volume = 100 if symbol in popular_symbols else 50
        volume_variation = np.random.uniform(0.5, 2.0)
        mentions_count = int(base_volume * volume_variation)
        
        # Calculate bullish/bearish ratios
        if base_sentiment > 0:
            bullish_ratio = 0.5 + (base_sentiment * 0.3)
            bearish_ratio = 1 - bullish_ratio
        else:
            bearish_ratio = 0.5 + (abs(base_sentiment) * 0.3)
            bullish_ratio = 1 - bearish_ratio
        
        return SocialSentiment(
            timestamp=datetime.now(TIMEZONE),
            platform='twitter',  # In production, would have multiple platforms
            symbol=symbol,
            sentiment_score=base_sentiment,
            volume_score=min(mentions_count / 200, 1.0),  # Normalize to [0, 1]
            mentions_count=mentions_count,
            bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio
        )

class AlternativeDataManager:
    """Main manager for all alternative data sources"""
    
    def __init__(self):
        self.logger = logging.getLogger('AlternativeDataManager')
        self.news_provider = NewsDataProvider()
        self.economic_provider = EconomicCalendarProvider()
        self.social_provider = SocialSentimentProvider()
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize alternative data database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # News events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    source TEXT,
                    symbols_mentioned TEXT,
                    sentiment_score REAL,
                    impact_score REAL,
                    category TEXT,
                    url TEXT
                )
            ''')
            
            # Economic events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS economic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    country TEXT,
                    event_name TEXT,
                    importance TEXT,
                    actual_value REAL,
                    forecast_value REAL,
                    previous_value REAL,
                    currency TEXT,
                    impact_score REAL
                )
            ''')
            
            # Social sentiment table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS social_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    platform TEXT,
                    symbol TEXT,
                    sentiment_score REAL,
                    volume_score REAL,
                    mentions_count INTEGER,
                    bullish_ratio REAL,
                    bearish_ratio REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing alternative data database: {e}")
    
    async def fetch_all_alternative_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch all alternative data for given symbols"""
        try:
            # Fetch data from all sources concurrently
            news_task = self.news_provider.fetch_financial_news(symbols)
            economic_task = self.economic_provider.fetch_economic_events()
            social_task = self.social_provider.fetch_social_sentiment(symbols)
            
            news_events, economic_events, social_sentiment = await asyncio.gather(
                news_task, economic_task, social_task
            )
            
            # Store in database
            await self._store_news_events(news_events)
            await self._store_economic_events(economic_events)
            await self._store_social_sentiment(social_sentiment)
            
            return {
                'news_events': news_events,
                'economic_events': economic_events,
                'social_sentiment': social_sentiment,
                'summary': self._generate_data_summary(news_events, economic_events, social_sentiment)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching alternative data: {e}")
            return {}
    
    async def _store_news_events(self, news_events: List[NewsEvent]):
        """Store news events in database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            for event in news_events:
                cursor.execute('''
                    INSERT INTO news_events 
                    (timestamp, title, content, source, symbols_mentioned, 
                     sentiment_score, impact_score, category, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(),
                    event.title,
                    event.content,
                    event.source,
                    json.dumps(event.symbols_mentioned),
                    event.sentiment_score,
                    event.impact_score,
                    event.category,
                    event.url
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing news events: {e}")
    
    async def _store_economic_events(self, economic_events: List[EconomicEvent]):
        """Store economic events in database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            for event in economic_events:
                cursor.execute('''
                    INSERT INTO economic_events 
                    (timestamp, country, event_name, importance, actual_value,
                     forecast_value, previous_value, currency, impact_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(),
                    event.country,
                    event.event_name,
                    event.importance,
                    event.actual_value,
                    event.forecast_value,
                    event.previous_value,
                    event.currency,
                    event.impact_score
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing economic events: {e}")
    
    async def _store_social_sentiment(self, social_sentiment: List[SocialSentiment]):
        """Store social sentiment in database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            for sentiment in social_sentiment:
                cursor.execute('''
                    INSERT INTO social_sentiment 
                    (timestamp, platform, symbol, sentiment_score, volume_score,
                     mentions_count, bullish_ratio, bearish_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sentiment.timestamp.isoformat(),
                    sentiment.platform,
                    sentiment.symbol,
                    sentiment.sentiment_score,
                    sentiment.volume_score,
                    sentiment.mentions_count,
                    sentiment.bullish_ratio,
                    sentiment.bearish_ratio
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing social sentiment: {e}")
    
    def _generate_data_summary(self, news_events: List[NewsEvent],
                             economic_events: List[EconomicEvent],
                             social_sentiment: List[SocialSentiment]) -> Dict[str, Any]:
        """Generate summary of alternative data"""
        try:
            summary = {
                'data_freshness': datetime.now(TIMEZONE).isoformat(),
                'total_news_events': len(news_events),
                'total_economic_events': len(economic_events),
                'total_social_signals': len(social_sentiment)
            }
            
            # News analysis
            if news_events:
                avg_sentiment = np.mean([event.sentiment_score for event in news_events])
                high_impact_news = [event for event in news_events if event.impact_score > 0.5]
                
                summary.update({
                    'news_avg_sentiment': avg_sentiment,
                    'high_impact_news_count': len(high_impact_news),
                    'news_categories': list(set(event.category for event in news_events))
                })
            
            # Economic events analysis
            if economic_events:
                high_importance_events = [event for event in economic_events if event.importance == 'High']
                upcoming_events_24h = [
                    event for event in economic_events 
                    if event.timestamp <= datetime.now(TIMEZONE) + timedelta(hours=24)
                ]
                
                summary.update({
                    'high_importance_events_count': len(high_importance_events),
                    'upcoming_events_24h': len(upcoming_events_24h),
                    'affected_currencies': list(set(event.currency for event in economic_events))
                })
            
            # Social sentiment analysis
            if social_sentiment:
                avg_sentiment = np.mean([sentiment.sentiment_score for sentiment in social_sentiment])
                total_mentions = sum(sentiment.mentions_count for sentiment in social_sentiment)
                
                summary.update({
                    'social_avg_sentiment': avg_sentiment,
                    'total_social_mentions': total_mentions,
                    'most_mentioned_symbols': self._get_most_mentioned_symbols(social_sentiment)
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating data summary: {e}")
            return {}
    
    def _get_most_mentioned_symbols(self, social_sentiment: List[SocialSentiment]) -> List[str]:
        """Get most mentioned symbols in social media"""
        symbol_mentions = {}
        
        for sentiment in social_sentiment:
            symbol = sentiment.symbol
            if symbol not in symbol_mentions:
                symbol_mentions[symbol] = 0
            symbol_mentions[symbol] += sentiment.mentions_count
        
        # Sort by mentions and return top 5
        sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_symbols[:5]]
    
    def get_sentiment_features(self, symbol: str, hours_back: int = 24) -> Dict[str, float]:
        """Get sentiment-based features for a symbol"""
        try:
            features = {}
            cutoff_time = datetime.now(TIMEZONE) - timedelta(hours=hours_back)
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # News sentiment features
            cursor.execute('''
                SELECT sentiment_score, impact_score FROM news_events 
                WHERE timestamp >= ? AND symbols_mentioned LIKE ?
            ''', (cutoff_time.isoformat(), f'%{symbol}%'))
            
            news_results = cursor.fetchall()
            if news_results:
                news_sentiments = [row[0] for row in news_results]
                news_impacts = [row[1] for row in news_results]
                
                features.update({
                    'news_sentiment_avg': np.mean(news_sentiments),
                    'news_sentiment_std': np.std(news_sentiments),
                    'news_impact_avg': np.mean(news_impacts),
                    'news_count': len(news_results),
                    'positive_news_ratio': sum(1 for s in news_sentiments if s > 0) / len(news_sentiments)
                })
            
            # Social sentiment features
            cursor.execute('''
                SELECT sentiment_score, volume_score, mentions_count, bullish_ratio 
                FROM social_sentiment 
                WHERE timestamp >= ? AND symbol = ?
            ''', (cutoff_time.isoformat(), symbol))
            
            social_results = cursor.fetchall()
            if social_results:
                social_sentiments = [row[0] for row in social_results]
                volume_scores = [row[1] for row in social_results]
                mentions = [row[2] for row in social_results]
                bullish_ratios = [row[3] for row in social_results]
                
                features.update({
                    'social_sentiment_avg': np.mean(social_sentiments),
                    'social_volume_avg': np.mean(volume_scores),
                    'social_mentions_total': sum(mentions),
                    'social_bullish_ratio': np.mean(bullish_ratios)
                })
            
            # Economic events features (for currency)
            currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
            cursor.execute('''
                SELECT impact_score, importance FROM economic_events 
                WHERE timestamp >= ? AND currency = ?
            ''', (cutoff_time.isoformat(), currency))
            
            economic_results = cursor.fetchall()
            if economic_results:
                impact_scores = [row[0] for row in economic_results]
                high_importance_count = sum(1 for row in economic_results if row[1] == 'High')
                
                features.update({
                    'economic_impact_avg': np.mean(impact_scores),
                    'economic_events_count': len(economic_results),
                    'high_importance_events': high_importance_count
                })
            
            conn.close()
            
            # Fill missing features with defaults
            default_features = {
                'news_sentiment_avg': 0.0,
                'news_sentiment_std': 0.0,
                'news_impact_avg': 0.0,
                'news_count': 0,
                'positive_news_ratio': 0.5,
                'social_sentiment_avg': 0.0,
                'social_volume_avg': 0.0,
                'social_mentions_total': 0,
                'social_bullish_ratio': 0.5,
                'economic_impact_avg': 0.0,
                'economic_events_count': 0,
                'high_importance_events': 0
            }
            
            for key, default_value in default_features.items():
                if key not in features:
                    features[key] = default_value
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment features: {e}")
            return {}
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming high-impact events"""
        try:
            future_time = datetime.now(TIMEZONE) + timedelta(hours=hours_ahead)
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT event_name, timestamp, country, currency, importance, impact_score
                FROM economic_events 
                WHERE timestamp <= ? AND timestamp > ?
                ORDER BY impact_score DESC, timestamp ASC
            ''', (future_time.isoformat(), datetime.now(TIMEZONE).isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            events = []
            for row in results:
                events.append({
                    'event_name': row[0],
                    'timestamp': row[1],
                    'country': row[2],
                    'currency': row[3],
                    'importance': row[4],
                    'impact_score': row[5]
                })
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting upcoming events: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old alternative data"""
        try:
            cutoff_date = (datetime.now(TIMEZONE) - timedelta(days=days_to_keep)).isoformat()
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Clean up old data
            cursor.execute('DELETE FROM news_events WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM economic_events WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM social_sentiment WHERE timestamp < ?', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old alternative data records")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up alternative data: {e}")
    
    def generate_market_context_report(self) -> str:
        """Generate a comprehensive market context report"""
        try:
            report = []
            report.append("=" * 50)
            report.append("MARKET CONTEXT REPORT")
            report.append("=" * 50)
            report.append("")
            
            # Recent news summary
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Get recent high-impact news
            recent_time = (datetime.now(TIMEZONE) - timedelta(hours=24)).isoformat()
            cursor.execute('''
                SELECT title, sentiment_score, impact_score, category 
                FROM news_events 
                WHERE timestamp >= ? AND impact_score > 0.5
                ORDER BY impact_score DESC LIMIT 5
            ''', (recent_time,))
            
            high_impact_news = cursor.fetchall()
            
            if high_impact_news:
                report.append("HIGH IMPACT NEWS (24H):")
                for title, sentiment, impact, category in high_impact_news:
                    sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                    report.append(f"• {title[:80]}...")
                    report.append(f"  Sentiment: {sentiment_label} ({sentiment:.2f}) | Impact: {impact:.2f} | Category: {category}")
                    report.append("")
            
            # Upcoming economic events
            future_time = (datetime.now(TIMEZONE) + timedelta(hours=24)).isoformat()
            cursor.execute('''
                SELECT event_name, timestamp, country, importance 
                FROM economic_events 
                WHERE timestamp <= ? AND timestamp > ?
                ORDER BY impact_score DESC LIMIT 5
            ''', (future_time, datetime.now(TIMEZONE).isoformat()))
            
            upcoming_events = cursor.fetchall()
            
            if upcoming_events:
                report.append("UPCOMING ECONOMIC EVENTS (24H):")
                for event_name, timestamp, country, importance in upcoming_events:
                    event_time = datetime.fromisoformat(timestamp).strftime("%m/%d %H:%M")
                    report.append(f"• {event_time} | {country} | {event_name} ({importance})")
                report.append("")
            
            # Social sentiment overview
            cursor.execute('''
                SELECT symbol, AVG(sentiment_score), SUM(mentions_count)
                FROM social_sentiment 
                WHERE timestamp >= ?
                GROUP BY symbol
                ORDER BY SUM(mentions_count) DESC LIMIT 5
            ''', (recent_time,))
            
            social_data = cursor.fetchall()
            
            if social_data:
                report.append("SOCIAL SENTIMENT OVERVIEW:")
                for symbol, avg_sentiment, total_mentions in social_data:
                    sentiment_label = "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"
                    report.append(f"• {symbol}: {sentiment_label} ({avg_sentiment:.2f}) | {total_mentions} mentions")
                report.append("")
            
            conn.close()
            
            report.append(f"Report generated: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating market context report: {e}")
            return "Error generating market context report"