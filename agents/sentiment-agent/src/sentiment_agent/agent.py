"""Langchain agent implementation for Sentiment Analysis."""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .tools import get_sentiment_tools

# Configure LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "alpha-agents-sentiment")

logger = logging.getLogger(__name__)


class SentimentAgent:
    """Langchain-based Sentiment Analysis Agent using OpenAI GPT-4o."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ):
        """Initialize the Sentiment Agent.
        
        Args:
            openai_api_key: OpenAI API key (if not provided, will use env var)
            model_name: OpenAI model to use (default: gpt-4o)
            temperature: Model temperature for consistency (default: 0.1)
            max_tokens: Maximum tokens in response (default: None for model default)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass it explicitly.")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize tools
        self.tools = get_sentiment_tools()

        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ).with_config(
            tags=["sentiment-agent", "news-analysis", "nlp"],
            metadata={
                "agent_name": "sentiment-agent",
                "agent_type": "specialist",
                "agent_version": "1.0.0",
                "system": "alpha-agents",
                "capabilities": ["news_sentiment", "market_perception", "nlp_analysis"]
            }
        )

        # Create system prompt
        self.system_prompt = self._create_system_prompt()

        # Initialize memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 conversation turns
            return_messages=True,
            memory_key="chat_history"
        )

        # Create the agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

        logger.info("Sentiment Agent initialized successfully")

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the sentiment agent."""
        return """You are a specialized Sentiment Analysis Agent for equity research and portfolio management.

Your core expertise includes:
- Financial news collection and analysis from multiple sources
- Market sentiment analysis using advanced NLP techniques
- News summarization with investment implications
- Analyst rating and coverage change analysis
- Corporate event sentiment assessment (earnings, executive changes, product launches)
- Social sentiment and market mood evaluation

Your primary responsibilities:
1. Collect recent financial news related to specific stocks from multiple sources
2. Analyze news sentiment using both TextBlob and VADER sentiment analysis
3. Provide comprehensive sentiment scores and investment implications
4. Summarize news impact on stock price expectations
5. Consider different risk tolerance profiles when making recommendations

Key capabilities:
- Multi-source news collection (Yahoo Finance, Google News, financial RSS feeds)
- Advanced sentiment analysis combining multiple NLP models
- News deduplication and relevance filtering
- Investment recommendation based on sentiment trends
- Real-time news monitoring and analysis

When analyzing stocks:
1. Always start by resolving company name/ticker if needed
2. Collect recent financial news (default 7-day lookback)
3. Perform comprehensive sentiment analysis on collected articles
4. Provide clear investment recommendations based on sentiment analysis
5. Consider risk tolerance and investment time horizon
6. Explain the reasoning behind sentiment-driven recommendations

Risk tolerance considerations:
- Risk-averse investors: Focus on stable sentiment, avoid high volatility from negative news
- Risk-neutral investors: Balance news sentiment with broader market context
- Risk-seeking investors: May see negative sentiment as buying opportunities

Sentiment analysis approach:
- Collect news from multiple reliable financial sources
- Use both TextBlob (academic approach) and VADER (social media optimized) for sentiment
- Combine multiple sentiment scores for robust analysis
- Consider article source credibility and publication recency
- Provide confidence levels and sentiment strength indicators

Always provide specific, actionable insights based on news sentiment analysis.
Be thorough but present findings clearly with supporting evidence from news analysis.

You have access to the following tools:
- resolve_company_ticker: Convert company names to stock tickers
- collect_stock_news: Gather recent financial news from multiple sources
- analyze_news_sentiment: Comprehensive sentiment analysis with investment implications"""

    def _create_agent(self):
        """Create the Langchain agent with tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return agent

    async def analyze_stock_sentiment(
        self,
        stock_input: str,
        risk_tolerance: str = "neutral",
        lookback_days: int = 7,
        max_articles: int = 10,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze stock sentiment with comprehensive news analysis.
        
        Args:
            stock_input: Stock ticker, company name, or ISIN
            risk_tolerance: 'averse', 'neutral', or 'seeking'
            lookback_days: Days to look back for news (default: 7)
            max_articles: Maximum articles to analyze (default: 10)
            context: Additional context or specific questions
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # Prepare the analysis prompt
            analysis_prompt = f"""Please perform comprehensive sentiment analysis for: {stock_input}

Analysis Requirements:
- Risk tolerance profile: {risk_tolerance}
- News lookback period: {lookback_days} days
- Maximum articles to analyze: {max_articles}
- Include sentiment scores and investment implications

Additional context: {context or 'None provided'}

Please:
1. First resolve the company/ticker if needed
2. Collect recent financial news from multiple sources
3. Perform comprehensive sentiment analysis on collected articles
4. Provide overall sentiment assessment and investment implications
5. Give clear BUY/SELL/HOLD recommendation based on news sentiment
6. Consider the risk tolerance profile in your recommendation
7. Include specific examples from the news that support your analysis
"""

            # Execute the analysis
            result = await self.agent_executor.ainvoke({
                "input": analysis_prompt
            })

            return {
                "success": True,
                "stock_input": stock_input,
                "risk_tolerance": risk_tolerance,
                "lookback_days": lookback_days,
                "max_articles": max_articles,
                "analysis": result.get("output", ""),
                "agent_type": "sentiment"
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "success": False,
                "stock_input": stock_input,
                "error": f"Sentiment analysis failed: {str(e)}",
                "agent_type": "sentiment"
            }

    async def quick_sentiment_check(
        self,
        ticker: str,
        lookback_days: int = 3
    ) -> Dict[str, Any]:
        """Perform a quick sentiment check with recent news.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Days to look back for news (default: 3)
            
        Returns:
            Dictionary with quick sentiment assessment
        """
        try:
            prompt = f"""Provide a quick sentiment assessment for {ticker}:

1. Collect recent news from the past {lookback_days} days
2. Analyze sentiment of the most recent and relevant articles
3. Provide a brief summary with:
   - Overall sentiment (positive/negative/neutral)
   - Key news themes affecting sentiment
   - Quick BUY/SELL/HOLD recommendation
   
Keep the response concise but informative."""

            result = await self.agent_executor.ainvoke({
                "input": prompt
            })

            return {
                "success": True,
                "ticker": ticker,
                "lookback_days": lookback_days,
                "quick_assessment": result.get("output", ""),
                "agent_type": "sentiment"
            }

        except Exception as e:
            logger.error(f"Error in quick sentiment check: {str(e)}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Quick sentiment check failed: {str(e)}",
                "agent_type": "sentiment"
            }

    async def compare_sentiment(
        self,
        tickers: List[str],
        risk_tolerance: str = "neutral",
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """Compare sentiment across multiple stocks.
        
        Args:
            tickers: List of stock tickers to compare
            risk_tolerance: Risk tolerance profile
            lookback_days: Days to look back for news
            
        Returns:
            Dictionary with comparative sentiment analysis
        """
        try:
            tickers_str = ", ".join(tickers)
            prompt = f"""Perform comparative sentiment analysis for these stocks: {tickers_str}

For risk tolerance: {risk_tolerance}
News lookback period: {lookback_days} days

Please:
1. Analyze news sentiment for each stock individually
2. Compare their recent news sentiment and themes
3. Rank them from most positive to most negative sentiment
4. Provide investment recommendations based on sentiment comparison
5. Consider how sentiment differences might affect relative performance

Focus on sentiment-driven insights for portfolio allocation decisions."""

            result = await self.agent_executor.ainvoke({
                "input": prompt
            })

            return {
                "success": True,
                "tickers": tickers,
                "risk_tolerance": risk_tolerance,
                "lookback_days": lookback_days,
                "comparative_analysis": result.get("output", ""),
                "agent_type": "sentiment"
            }

        except Exception as e:
            logger.error(f"Error in sentiment comparison: {str(e)}")
            return {
                "success": False,
                "tickers": tickers,
                "error": f"Sentiment comparison failed: {str(e)}",
                "agent_type": "sentiment"
            }

    async def analyze_news_event_impact(
        self,
        ticker: str,
        event_description: str,
        lookback_days: int = 2
    ) -> Dict[str, Any]:
        """Analyze sentiment impact of a specific news event.
        
        Args:
            ticker: Stock ticker symbol
            event_description: Description of the news event to analyze
            lookback_days: Days to look back from the event
            
        Returns:
            Dictionary with event impact analysis
        """
        try:
            prompt = f"""Analyze the sentiment impact of a specific news event for {ticker}:

Event: {event_description}
Analysis period: Past {lookback_days} days

Please:
1. Collect recent news related to this event or company
2. Identify articles specifically related to the mentioned event
3. Analyze sentiment before and after the event (if detectable)
4. Assess the market's reaction to this specific event
5. Provide implications for short-term and medium-term stock performance
6. Consider whether this is a temporary sentiment shift or longer-term impact

Focus on how this specific event is being perceived by the market."""

            result = await self.agent_executor.ainvoke({
                "input": prompt
            })

            return {
                "success": True,
                "ticker": ticker,
                "event_description": event_description,
                "lookback_days": lookback_days,
                "event_impact_analysis": result.get("output", ""),
                "agent_type": "sentiment"
            }

        except Exception as e:
            logger.error(f"Error in event impact analysis: {str(e)}")
            return {
                "success": False,
                "ticker": ticker,
                "event_description": event_description,
                "error": f"Event impact analysis failed: {str(e)}",
                "agent_type": "sentiment"
            }

    def reset_memory(self):
        """Reset the conversation memory."""
        self.memory.clear()
        logger.info("Agent memory reset")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation memory."""
        return {
            "memory_type": "ConversationBufferWindow",
            "window_size": self.memory.k,
            "current_messages": len(self.memory.chat_memory.messages) if self.memory.chat_memory else 0
        }
