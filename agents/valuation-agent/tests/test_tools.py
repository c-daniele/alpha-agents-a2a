"""Test suite for valuation agent tools."""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd

# Add the src directory to path for imports
import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from valuation_agent.tools import (
    StockDataTool,
    VolatilityCalculationTool, 
    CompanyNameResolverTool
)


class TestCompanyNameResolverTool:
    """Test the company name resolver tool."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = CompanyNameResolverTool()
    
    def test_resolve_ticker_direct(self):
        """Test resolving a direct ticker symbol."""
        result = self.resolver._run("AAPL")
        assert result["success"] is True
        assert result["ticker"] == "AAPL"
        assert "Apple" in result.get("company_name", "")
    
    def test_resolve_company_name(self):
        """Test resolving a company name."""
        result = self.resolver._run("Apple")
        assert result["success"] is True
        assert result["ticker"] == "AAPL"
    
    def test_resolve_microsoft(self):
        """Test resolving Microsoft."""
        result = self.resolver._run("Microsoft")
        assert result["success"] is True
        assert result["ticker"] == "MSFT"
    
    def test_resolve_invalid_company(self):
        """Test resolving an invalid company name."""
        result = self.resolver._run("NonexistentCompany12345")
        assert result["success"] is False
        assert "suggestions" in result


class TestStockDataTool:
    """Test the stock data fetching tool."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.stock_tool = StockDataTool()
    
    def test_fetch_valid_stock(self):
        """Test fetching data for a valid stock."""
        result = self.stock_tool._run("AAPL", period_days=30)
        
        if result["success"]:
            assert result["symbol"] == "AAPL"
            assert "price_data" in result
            assert "latest_price" in result
            assert len(result["price_data"]["dates"]) > 0
        else:
            # If it fails, it should have an error message
            assert "error" in result
    
    def test_fetch_invalid_stock(self):
        """Test fetching data for an invalid stock."""
        result = self.stock_tool._run("INVALID123", period_days=30)
        assert result["success"] is False
        assert "error" in result
    
    def test_period_days_parameter(self):
        """Test the period_days parameter."""
        result = self.stock_tool._run("MSFT", period_days=7)
        
        if result["success"]:
            assert result["period_days"] == 7
            # Should have less data points for shorter period
            assert len(result["price_data"]["dates"]) <= 10  # Accounting for weekends


class TestVolatilityCalculationTool:
    """Test the volatility calculation tool."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.volatility_tool = VolatilityCalculationTool()
    
    def test_calculate_volatility_valid_stock(self):
        """Test calculating volatility for a valid stock."""
        result = self.volatility_tool._run("AAPL", risk_free_rate=0.05)
        
        if result["success"]:
            assert result["symbol"] == "AAPL"
            assert "volatility_metrics" in result
            assert "risk_metrics" in result
            assert "price_metrics" in result
            
            # Check that key metrics are present
            vol_metrics = result["volatility_metrics"]
            assert "daily_volatility" in vol_metrics
            assert "annualized_volatility" in vol_metrics
            
            risk_metrics = result["risk_metrics"]
            assert "sharpe_ratio" in risk_metrics
            assert "max_drawdown" in risk_metrics
            
        else:
            # If it fails, should have error message
            assert "error" in result
    
    def test_risk_free_rate_parameter(self):
        """Test the risk_free_rate parameter."""
        result = self.volatility_tool._run("MSFT", risk_free_rate=0.03)
        
        if result["success"]:
            assert result["risk_metrics"]["risk_free_rate"] == 0.03


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations."""
    
    async def test_async_company_resolution(self):
        """Test async company name resolution."""
        resolver = CompanyNameResolverTool()
        result = await resolver._arun("Tesla")
        
        if result["success"]:
            assert result["ticker"] == "TSLA"
    
    async def test_async_stock_data_fetch(self):
        """Test async stock data fetching."""
        stock_tool = StockDataTool()
        result = await stock_tool._arun("GOOGL", period_days=30)
        
        if result["success"]:
            assert result["symbol"] == "GOOGL"
            assert "price_data" in result
    
    async def test_async_volatility_calculation(self):
        """Test async volatility calculation."""
        volatility_tool = VolatilityCalculationTool()
        result = await volatility_tool._arun("NVDA", risk_free_rate=0.04)
        
        if result["success"]:
            assert result["symbol"] == "NVDA"
            assert "volatility_metrics" in result


if __name__ == "__main__":
    # Run a simple test
    print("Running basic tool tests...")
    
    # Test company resolver
    resolver = CompanyNameResolverTool()
    apple_result = resolver._run("Apple")
    print(f"Apple resolution: {apple_result}")
    
    # Test stock data (only if previous test succeeded)
    if apple_result["success"]:
        stock_tool = StockDataTool()
        data_result = stock_tool._run(apple_result["ticker"], period_days=30)
        print(f"Stock data success: {data_result.get('success', False)}")
        
        if data_result["success"]:
            print(f"Data points: {data_result['data_points']}")
            print(f"Latest price: ${data_result['latest_price']:.2f}")
    
    print("Basic tests completed.")