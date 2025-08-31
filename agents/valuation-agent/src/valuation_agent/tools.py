"""Tools for the Valuation Agent - Yahoo Finance data fetching and financial calculations."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Trading days per year constant
TRADING_DAYS_PER_YEAR = 252


class StockDataInput(BaseModel):
    """Input schema for stock data fetching."""
    
    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    period_days: int = Field(
        default=365, 
        description="Number of days of historical data to fetch"
    )


class VolatilityCalculationInput(BaseModel):
    """Input schema for volatility and return calculations."""
    
    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    risk_free_rate: float = Field(
        default=0.05, 
        description="Risk-free rate for Sharpe ratio calculation (default 5%)"
    )


class StockDataTool(BaseTool):
    """Tool to fetch historical stock price and volume data from Yahoo Finance."""
    
    name: str = "fetch_stock_data"
    description: str = (
        "Fetches historical stock price and volume data from Yahoo Finance. "
        "Input should be a stock symbol (ticker) and optional number of days. "
        "Returns OHLCV data with dates."
    )
    args_schema: type[BaseModel] = StockDataInput

    def _run(self, symbol: str, period_days: int = 365) -> Dict[str, Any]:
        """Fetch stock data synchronously."""
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(
                start=start_date, 
                end=end_date, 
                auto_adjust=True,
                back_adjust=True
            )
            
            if data.empty:
                return {
                    "success": False,
                    "error": f"No data found for symbol {symbol}",
                    "symbol": symbol
                }
            
            # Convert to dictionary format
            result = {
                "success": True,
                "symbol": symbol.upper(),
                "period_days": period_days,
                "data_points": len(data),
                "start_date": data.index[0].strftime("%Y-%m-%d"),
                "end_date": data.index[-1].strftime("%Y-%m-%d"),
                "latest_price": float(data['Close'].iloc[-1]),
                "price_data": {
                    "dates": data.index.strftime("%Y-%m-%d").tolist(),
                    "open": data['Open'].tolist(),
                    "high": data['High'].tolist(),
                    "low": data['Low'].tolist(),
                    "close": data['Close'].tolist(),
                    "volume": data['Volume'].tolist()
                }
            }
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to fetch data for {symbol}: {str(e)}",
                "symbol": symbol
            }

    async def _arun(self, symbol: str, period_days: int = 365) -> Dict[str, Any]:
        """Fetch stock data asynchronously."""
        return await asyncio.to_thread(self._run, symbol, period_days)


class VolatilityCalculationTool(BaseTool):
    """Tool to calculate volatility, expected returns, and risk metrics."""
    
    name: str = "calculate_volatility_metrics"
    description: str = (
        "Calculates comprehensive volatility and risk metrics for a stock including "
        "daily/annualized volatility, expected returns, Sharpe ratio, maximum drawdown, "
        "and other risk measures. Requires stock symbol as input."
    )
    args_schema: type[BaseModel] = VolatilityCalculationInput

    def _run(self, symbol: str, risk_free_rate: float = 0.05) -> Dict[str, Any]:
        """Calculate volatility and risk metrics synchronously."""
        try:
            # First fetch the data
            stock_tool = StockDataTool()
            stock_data = stock_tool._run(symbol)
            
            if not stock_data["success"]:
                return stock_data
            
            # Convert to pandas DataFrame for calculations
            price_data = stock_data["price_data"]
            df = pd.DataFrame({
                'Date': pd.to_datetime(price_data["dates"]),
                'Close': price_data["close"],
                'Volume': price_data["volume"]
            })
            df.set_index('Date', inplace=True)
            
            # Calculate daily returns
            df['Daily_Return'] = df['Close'].pct_change()
            df = df.dropna()
            
            if len(df) < 2:
                return {
                    "success": False,
                    "error": "Insufficient data for calculations",
                    "symbol": symbol
                }
            
            # Basic metrics
            daily_returns = df['Daily_Return']
            
            # Daily metrics
            mean_daily_return = daily_returns.mean()
            daily_volatility = daily_returns.std()
            
            # Calculate cumulative return for proper annualized return
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            cumulative_return = (end_price / start_price) - 1
            trading_days = len(df)
            
            # Annualized metrics
            annualized_return = ((1 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / trading_days)) - 1
            annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Sharpe ratio
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Maximum drawdown calculation
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Value at Risk (VaR) - 5% and 1%
            var_5 = np.percentile(daily_returns, 5)
            var_1 = np.percentile(daily_returns, 1)
            
            # Additional statistics
            skewness = daily_returns.skew()
            kurtosis = daily_returns.kurtosis()
            
            # Price performance metrics
            total_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            
            result = {
                "success": True,
                "symbol": symbol.upper(),
                "analysis_period": {
                    "start_date": df.index[0].strftime("%Y-%m-%d"),
                    "end_date": df.index[-1].strftime("%Y-%m-%d"),
                    "trading_days": len(df)
                },
                "price_metrics": {
                    "start_price": float(df['Close'].iloc[0]),
                    "end_price": float(df['Close'].iloc[-1]),
                    "total_return": float(total_return),
                    "annualized_return": float(annualized_return)
                },
                "volatility_metrics": {
                    "daily_volatility": float(daily_volatility),
                    "annualized_volatility": float(annualized_volatility),
                    "volatility_percentage": float(annualized_volatility * 100)
                },
                "risk_metrics": {
                    "sharpe_ratio": float(sharpe_ratio),
                    "max_drawdown": float(max_drawdown),
                    "max_drawdown_percentage": float(max_drawdown * 100),
                    "var_5_percent": float(var_5),
                    "var_1_percent": float(var_1),
                    "risk_free_rate": float(risk_free_rate)
                },
                "distribution_metrics": {
                    "mean_daily_return": float(mean_daily_return),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "positive_days": int((daily_returns > 0).sum()),
                    "negative_days": int((daily_returns < 0).sum())
                },
                "volume_metrics": {
                    "average_volume": float(df['Volume'].mean()),
                    "volume_volatility": float(df['Volume'].std()),
                    "latest_volume": float(df['Volume'].iloc[-1])
                }
            }
            
            logger.info(f"Successfully calculated metrics for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to calculate metrics for {symbol}: {str(e)}",
                "symbol": symbol
            }

    async def _arun(self, symbol: str, risk_free_rate: float = 0.05) -> Dict[str, Any]:
        """Calculate volatility and risk metrics asynchronously."""
        return await asyncio.to_thread(self._run, symbol, risk_free_rate)


class CompanyResolverInput(BaseModel):
    """Input schema for company name/ISIN resolution."""
    
    query: str = Field(description="Company name, ISIN, or potential ticker symbol to resolve")


class CompanyNameResolverTool(BaseTool):
    """Tool to resolve company names or ISIN to stock tickers."""
    
    name: str = "resolve_company_ticker"
    description: str = (
        "Resolves company names (e.g., 'Apple', 'Microsoft') or ISIN codes to stock tickers. "
        "Returns the ticker symbol that can be used for data fetching."
    )
    args_schema: type[BaseModel] = CompanyResolverInput

    def _run(self, query: str) -> Dict[str, Any]:
        """Resolve company name/ISIN to ticker symbol."""
        try:
            query = query.strip()
            
            # Common company name mappings
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
                
                # Other major companies
                "walmart": "WMT",
                "berkshire hathaway": "BRK-B",
                "johnson & johnson": "JNJ",
                "exxon mobil": "XOM",
                "unitedhealth": "UNH",
                "jpmorgan": "JPM",
                "jp morgan": "JPM",
                "procter & gamble": "PG",
                "visa": "V",
                "home depot": "HD",
                "mastercard": "MA",
                "coca cola": "KO",
                "coca-cola": "KO",
                "disney": "DIS",
                "walt disney": "DIS",
                "nike": "NKE",
                "intel": "INTC",
                "verizon": "VZ",
                "at&t": "T",
                "chevron": "CVX",
                "salesforce": "CRM",
                "adobe": "ADBE",
                "netflix": "NFLX",
                "paypal": "PYPL",
                "ibm": "IBM",
                "oracle": "ORCL",
                "cisco": "CSCO"
            }
            
            # Check if it's already a ticker (all caps, 1-5 characters)
            if query.isupper() and 1 <= len(query) <= 5 and query.isalpha():
                # Validate it exists by trying to fetch basic info
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
            
            # Check company name mappings
            query_lower = query.lower()
            if query_lower in company_mappings:
                ticker = company_mappings[query_lower]
                try:
                    # Validate the ticker
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
            
            # Try searching by company name using yfinance
            try:
                # Use yfinance search functionality (if available)
                # This is a fallback for less common companies

                logger.info(f"Attempting fallback search for: {query}")

                potential_ticker = query.upper()
                ticker = yf.Ticker(potential_ticker)
                info = ticker.info

                logger.info(f"Search attempt for '{query}': {info}")

                if info and 'symbol' in info:
                    return {
                        "success": True,
                        "query": query,
                        "ticker": potential_ticker,
                        "company_name": info.get('longName', 'Unknown'),
                        "resolution_method": "search_attempt"
                    }
            except:
                pass
            
            # If all else fails, return an error with suggestions
            return {
                "success": False,
                "query": query,
                "error": f"Could not resolve '{query}' to a valid stock ticker",
                "suggestions": [
                    "Try using the stock ticker directly (e.g., AAPL for Apple)",
                    "Check spelling of company name",
                    "Use common company names like 'Apple', 'Microsoft', 'Google', etc."
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
def get_valuation_tools() -> List[BaseTool]:
    """Get all valuation analysis tools."""
    return [
        CompanyNameResolverTool(),
        StockDataTool(),
        VolatilityCalculationTool()
    ]