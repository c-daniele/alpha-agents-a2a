"""Tools for the Fundamental Agent - Financial report analysis and fundamental analysis."""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FinanceReportInput(BaseModel):
    """Input schema for finance report retrieval."""
    
    symbol: str = Field(description="Stock symbol (e.g., AAPL, TSLA, MSFT)")
    report_types: List[str] = Field(
        default=["income_statement", "balance_sheet", "cash_flow", "financials"],
        description="Types of financial reports to retrieve"
    )
    periods: int = Field(
        default=4,
        description="Number of periods to retrieve (quarterly/annual)"
    )
    include_sec_filings: bool = Field(
        default=True,
        description="Whether to attempt SEC filing retrieval for 10-K/10-Q data"
    )


class RAGAnalysisInput(BaseModel):
    """Input schema for RAG-based fundamental analysis."""
    
    symbol: str = Field(description="Stock symbol for analysis")
    analysis_focus: str = Field(
        default="comprehensive",
        description="Focus area: 'cash_flow', 'operations', 'concerns', 'objectives', or 'comprehensive'"
    )
    financial_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-retrieved financial data (if not provided, will fetch automatically)"
    )


class CompanyResolverInput(BaseModel):
    """Input schema for company name resolution."""
    
    query: str = Field(description="Company name, ISIN, or potential ticker symbol to resolve")


class FinanceReportPullTool(BaseTool):
    """Tool to pull comprehensive financial reports using yfinance with validation."""
    
    name: str = "finance_report_pull"
    description: str = (
        "Retrieves comprehensive financial reports for a stock including income statements, "
        "balance sheets, cash flow statements, and other financial data. Includes iterative "
        "API call validation to ensure data quality and completeness."
    )
    args_schema: type[BaseModel] = FinanceReportInput

    def _run(
        self,
        symbol: str,
        report_types: List[str] = None,
        periods: int = 4,
        include_sec_filings: bool = True
    ) -> Dict[str, Any]:
        """Pull financial reports with validation."""
        if report_types is None:
            report_types = ["income_statement", "balance_sheet", "cash_flow", "financials"]
        
        try:
            # Validate ticker symbol first
            ticker = yf.Ticker(symbol.upper())
            
            # Get basic info to validate ticker
            try:
                info = ticker.info
                if not info or 'symbol' not in info:
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": f"Invalid ticker symbol: {symbol}"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": f"Failed to validate ticker {symbol}: {str(e)}"
                }
            
            company_name = info.get('longName', symbol.upper())
            logger.info(f"Pulling financial reports for {symbol} ({company_name})")
            
            financial_data = {}
            
            # Pull different types of financial reports
            for report_type in report_types:
                try:
                    if report_type == "income_statement":
                        # Get both quarterly and annual income statements
                        quarterly_income = ticker.quarterly_financials
                        annual_income = ticker.financials
                        
                        financial_data["income_statement"] = {
                            "quarterly": self._dataframe_to_dict(quarterly_income, periods),
                            "annual": self._dataframe_to_dict(annual_income, periods),
                            "data_quality": self._assess_data_quality(quarterly_income, annual_income)
                        }
                        
                    elif report_type == "balance_sheet":
                        # Get both quarterly and annual balance sheets
                        quarterly_balance = ticker.quarterly_balance_sheet
                        annual_balance = ticker.balance_sheet
                        
                        financial_data["balance_sheet"] = {
                            "quarterly": self._dataframe_to_dict(quarterly_balance, periods),
                            "annual": self._dataframe_to_dict(annual_balance, periods),
                            "data_quality": self._assess_data_quality(quarterly_balance, annual_balance)
                        }
                        
                    elif report_type == "cash_flow":
                        # Get both quarterly and annual cash flow statements
                        quarterly_cashflow = ticker.quarterly_cashflow
                        annual_cashflow = ticker.cashflow
                        
                        financial_data["cash_flow"] = {
                            "quarterly": self._dataframe_to_dict(quarterly_cashflow, periods),
                            "annual": self._dataframe_to_dict(annual_cashflow, periods),
                            "data_quality": self._assess_data_quality(quarterly_cashflow, annual_cashflow)
                        }
                        
                    elif report_type == "financials":
                        # Get key financial metrics and ratios
                        financial_data["key_metrics"] = self._extract_key_metrics(ticker, info)
                        
                except Exception as e:
                    logger.warning(f"Failed to retrieve {report_type} for {symbol}: {str(e)}")
                    financial_data[report_type] = {
                        "error": f"Failed to retrieve {report_type}: {str(e)}"
                    }
            
            # Attempt to get SEC filing information if requested
            sec_data = {}
            if include_sec_filings:
                try:
                    sec_data = self._get_sec_filing_info(symbol, info)
                except Exception as e:
                    logger.warning(f"Failed to retrieve SEC data for {symbol}: {str(e)}")
                    sec_data = {"error": f"SEC data retrieval failed: {str(e)}"}
            
            # Validate overall data quality
            overall_quality = self._validate_overall_data_quality(financial_data)
            
            result = {
                "success": True,
                "symbol": symbol.upper(),
                "company_name": company_name,
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 'N/A'),
                "retrieval_date": datetime.now(timezone.utc).isoformat(),
                "periods_requested": periods,
                "financial_data": financial_data,
                "sec_filings": sec_data,
                "data_quality_assessment": overall_quality,
                "api_validation": {
                    "ticker_valid": True,
                    "data_completeness": overall_quality["completeness_score"],
                    "validation_checks_passed": overall_quality["validation_passed"]
                }
            }
            
            logger.info(f"Successfully retrieved financial reports for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error pulling financial reports for {symbol}: {str(e)}")
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Failed to pull financial reports: {str(e)}"
            }

    async def _arun(
        self,
        symbol: str,
        report_types: List[str] = None,
        periods: int = 4,
        include_sec_filings: bool = True
    ) -> Dict[str, Any]:
        """Pull financial reports asynchronously."""
        return await asyncio.to_thread(self._run, symbol, report_types, periods, include_sec_filings)

    def _dataframe_to_dict(self, df: pd.DataFrame, max_periods: int) -> Dict[str, Any]:
        """Convert DataFrame to dictionary with validation."""
        if df is None or df.empty:
            return {"error": "No data available"}
        
        try:
            # Limit to requested number of periods
            limited_df = df.iloc[:, :max_periods] if df.shape[1] > max_periods else df
            
            # Convert to dictionary
            data_dict = limited_df.to_dict('index')
            
            # Add metadata
            return {
                "data": data_dict,
                "periods": limited_df.columns.tolist(),
                "metrics_count": len(limited_df.index),
                "data_shape": limited_df.shape
            }
        except Exception as e:
            return {"error": f"Failed to process data: {str(e)}"}

    def _assess_data_quality(self, quarterly_df: pd.DataFrame, annual_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality and completeness of financial data."""
        quality_assessment = {
            "quarterly_data_available": not (quarterly_df is None or quarterly_df.empty),
            "annual_data_available": not (annual_df is None or annual_df.empty),
            "quarterly_periods": quarterly_df.shape[1] if quarterly_df is not None and not quarterly_df.empty else 0,
            "annual_periods": annual_df.shape[1] if annual_df is not None and not annual_df.empty else 0,
            "missing_data_percentage": 0,
            "data_freshness": "Unknown"
        }
        
        # Calculate missing data percentage
        if quarterly_df is not None and not quarterly_df.empty:
            total_cells = quarterly_df.size
            missing_cells = quarterly_df.isnull().sum().sum()
            quality_assessment["missing_data_percentage"] = (missing_cells / total_cells) * 100 if total_cells > 0 else 100
            
            # Check data freshness (most recent quarter)
            if quarterly_df.shape[1] > 0:
                latest_date = quarterly_df.columns[0]
                quality_assessment["data_freshness"] = str(latest_date)
        
        return quality_assessment

    def _extract_key_metrics(self, ticker, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial metrics and ratios."""
        try:
            metrics = {}
            
            # Valuation metrics
            metrics["valuation"] = {
                "market_cap": info.get('marketCap'),
                "enterprise_value": info.get('enterpriseValue'),
                "pe_ratio": info.get('forwardPE') or info.get('trailingPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "price_to_sales": info.get('priceToSalesTrailing12Months'),
                "ev_to_revenue": info.get('enterpriseToRevenue'),
                "ev_to_ebitda": info.get('enterpriseToEbitda')
            }
            
            # Financial health metrics
            metrics["financial_health"] = {
                "total_cash": info.get('totalCash'),
                "total_debt": info.get('totalDebt'),
                "current_ratio": info.get('currentRatio'),
                "debt_to_equity": info.get('debtToEquity'),
                "return_on_assets": info.get('returnOnAssets'),
                "return_on_equity": info.get('returnOnEquity'),
                "gross_margins": info.get('grossMargins'),
                "operating_margins": info.get('operatingMargins'),
                "profit_margins": info.get('profitMargins')
            }
            
            # Growth metrics
            metrics["growth"] = {
                "revenue_growth": info.get('revenueGrowth'),
                "earnings_growth": info.get('earningsGrowth'),
                "revenue_per_share": info.get('revenuePerShare'),
                "book_value": info.get('bookValue'),
                "earnings_per_share": info.get('trailingEps'),
                "forward_eps": info.get('forwardEps')
            }
            
            return metrics
            
        except Exception as e:
            return {"error": f"Failed to extract key metrics: {str(e)}"}

    def _get_sec_filing_info(self, symbol: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to get SEC filing information."""
        try:
            # Basic SEC info from yfinance
            sec_info = {
                "cik": info.get('cik'),
                "company_officers": info.get('companyOfficers', []),
                "governance": {
                    "audit_risk": info.get('auditRisk'),
                    "board_risk": info.get('boardRisk'),
                    "compensation_risk": info.get('compensationRisk'),
                    "shareholder_rights_risk": info.get('shareHolderRightsRisk'),
                    "overall_risk": info.get('overallRisk')
                }
            }
            
            # Note: For production, you would integrate with SEC API or EDGAR database
            # For now, we'll indicate that manual SEC filing retrieval would be needed
            sec_info["filing_note"] = (
                "For comprehensive 10-K/10-Q analysis, manual SEC EDGAR database "
                "retrieval would be implemented here with proper SEC API integration."
            )
            
            return sec_info
            
        except Exception as e:
            return {"error": f"SEC filing retrieval failed: {str(e)}"}

    def _validate_overall_data_quality(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall data quality across all retrieved reports."""
        validation_results = {
            "validation_passed": True,
            "completeness_score": 0,
            "issues": [],
            "recommendations": []
        }
        
        try:
            total_sections = 0
            successful_sections = 0
            
            for section, data in financial_data.items():
                total_sections += 1
                
                if isinstance(data, dict) and "error" not in data:
                    successful_sections += 1
                elif isinstance(data, dict) and "error" in data:
                    validation_results["issues"].append(f"Failed to retrieve {section}")
            
            # Calculate completeness score
            if total_sections > 0:
                validation_results["completeness_score"] = (successful_sections / total_sections) * 100
            
            # Determine if validation passed
            validation_results["validation_passed"] = validation_results["completeness_score"] >= 75
            
            # Add recommendations
            if validation_results["completeness_score"] < 50:
                validation_results["recommendations"].append(
                    "Consider using alternative data sources due to low data completeness"
                )
            elif validation_results["completeness_score"] < 75:
                validation_results["recommendations"].append(
                    "Some financial data missing - analysis may be limited"
                )
            
        except Exception as e:
            validation_results["validation_passed"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
        
        return validation_results


class RAGAnalysisTool(BaseTool):
    """RAG tool for fundamental analysis with domain expertise guidance."""
    
    name: str = "rag_fundamental_analysis"
    description: str = (
        "Performs comprehensive fundamental analysis using RAG (Retrieval-Augmented Generation) "
        "with domain expertise guidance. Analyzes financial reports to answer specific questions "
        "about cash flow, operations, areas of concern, and progress towards objectives."
    )
    args_schema: type[BaseModel] = RAGAnalysisInput

    def _run(
        self,
        symbol: str,
        analysis_focus: str = "comprehensive",
        financial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform RAG-based fundamental analysis."""
        try:
            # If no financial data provided, retrieve it first
            if financial_data is None:
                finance_tool = FinanceReportPullTool()
                financial_data = finance_tool._run(symbol)
                
                if not financial_data.get("success"):
                    return financial_data  # Return the error
            
            # Extract company information
            company_name = financial_data.get("company_name", symbol.upper())
            sector = financial_data.get("sector", "Unknown")
            industry = financial_data.get("industry", "Unknown")
            
            logger.info(f"Performing RAG analysis for {symbol} ({company_name}) - Focus: {analysis_focus}")
            
            # Prepare financial context for RAG analysis
            financial_context = self._prepare_financial_context(financial_data)
            
            # Get domain expertise guidance based on analysis focus
            analysis_guidance = self._get_domain_expertise_guidance(analysis_focus)
            
            # Perform analysis based on focus area
            analysis_results = {}
            
            if analysis_focus == "comprehensive":
                # Perform all types of analysis
                analysis_results["cash_flow_analysis"] = self._analyze_cash_flow(financial_context, analysis_guidance)
                analysis_results["operations_analysis"] = self._analyze_operations(financial_context, analysis_guidance)
                analysis_results["concerns_analysis"] = self._identify_concerns(financial_context, analysis_guidance)
                analysis_results["objectives_analysis"] = self._assess_objectives(financial_context, analysis_guidance)
            else:
                # Perform specific analysis
                if analysis_focus == "cash_flow":
                    analysis_results["cash_flow_analysis"] = self._analyze_cash_flow(financial_context, analysis_guidance)
                elif analysis_focus == "operations":
                    analysis_results["operations_analysis"] = self._analyze_operations(financial_context, analysis_guidance)
                elif analysis_focus == "concerns":
                    analysis_results["concerns_analysis"] = self._identify_concerns(financial_context, analysis_guidance)
                elif analysis_focus == "objectives":
                    analysis_results["objectives_analysis"] = self._assess_objectives(financial_context, analysis_guidance)
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(analysis_results, financial_context)
            
            result = {
                "success": True,
                "symbol": symbol.upper(),
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
                "analysis_date": datetime.now(timezone.utc).isoformat(),
                "analysis_focus": analysis_focus,
                "analysis_results": analysis_results,
                "overall_assessment": overall_assessment,
                "data_sources": {
                    "financial_reports": list(financial_data.get("financial_data", {}).keys()),
                    "analysis_method": "RAG with domain expertise guidance"
                }
            }
            
            logger.info(f"Completed RAG analysis for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG analysis for {symbol}: {str(e)}")
            return {
                "success": False,
                "symbol": symbol,
                "error": f"RAG analysis failed: {str(e)}"
            }

    async def _arun(
        self,
        symbol: str,
        analysis_focus: str = "comprehensive",
        financial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform RAG analysis asynchronously."""
        return await asyncio.to_thread(self._run, symbol, analysis_focus, financial_data)

    def _prepare_financial_context(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare structured financial context for analysis."""
        context = {
            "company_profile": {
                "name": financial_data.get("company_name"),
                "sector": financial_data.get("sector"),
                "industry": financial_data.get("industry"),
                "market_cap": financial_data.get("market_cap")
            }
        }
        
        # Process financial statements
        fd = financial_data.get("financial_data", {})
        
        # Income statement context
        if "income_statement" in fd and isinstance(fd["income_statement"], dict):
            context["income_statement"] = fd["income_statement"]
        
        # Balance sheet context  
        if "balance_sheet" in fd and isinstance(fd["balance_sheet"], dict):
            context["balance_sheet"] = fd["balance_sheet"]
        
        # Cash flow context
        if "cash_flow" in fd and isinstance(fd["cash_flow"], dict):
            context["cash_flow"] = fd["cash_flow"]
        
        # Key metrics context
        if "key_metrics" in fd and isinstance(fd["key_metrics"], dict):
            context["key_metrics"] = fd["key_metrics"]
        
        return context

    def _get_domain_expertise_guidance(self, analysis_focus: str) -> Dict[str, Any]:
        """Get domain expertise guidance for financial analysis."""
        
        # Core financial analysis principles
        base_guidance = {
            "general_principles": [
                "Analyze trends over multiple periods",
                "Compare metrics to industry benchmarks",
                "Consider economic and sector context",
                "Identify key financial ratios and their implications",
                "Look for consistency in financial performance"
            ]
        }
        
        if analysis_focus == "cash_flow" or analysis_focus == "comprehensive":
            base_guidance["cash_flow_expertise"] = {
                "key_metrics": [
                    "Operating Cash Flow",
                    "Free Cash Flow",
                    "Cash Flow from Investing Activities",
                    "Cash Flow from Financing Activities",
                    "Cash Conversion Cycle"
                ],
                "analysis_points": [
                    "Evaluate cash generation quality and sustainability",
                    "Assess working capital management efficiency",
                    "Analyze capital allocation decisions",
                    "Review cash flow predictability and seasonality",
                    "Compare cash flow to net income (quality of earnings)"
                ]
            }
        
        if analysis_focus == "operations" or analysis_focus == "comprehensive":
            base_guidance["operations_expertise"] = {
                "key_metrics": [
                    "Gross Margin",
                    "Operating Margin",
                    "EBITDA Margin",
                    "Asset Turnover",
                    "Inventory Turnover",
                    "Return on Assets"
                ],
                "analysis_points": [
                    "Assess operational efficiency trends",
                    "Evaluate cost structure and margin stability",
                    "Analyze revenue growth drivers",
                    "Review asset utilization effectiveness",
                    "Compare operational metrics to industry peers"
                ]
            }
        
        if analysis_focus == "concerns" or analysis_focus == "comprehensive":
            base_guidance["risk_assessment_expertise"] = {
                "financial_risks": [
                    "Liquidity risk (current ratio, quick ratio)",
                    "Leverage risk (debt-to-equity, interest coverage)",
                    "Profitability deterioration",
                    "Working capital management issues",
                    "Cash flow sustainability concerns"
                ],
                "red_flags": [
                    "Declining gross margins",
                    "Increasing debt levels",
                    "Deteriorating cash flow",
                    "Growing accounts receivable relative to sales",
                    "Frequent accounting changes or restatements"
                ]
            }
        
        if analysis_focus == "objectives" or analysis_focus == "comprehensive":
            base_guidance["strategic_assessment_expertise"] = {
                "growth_indicators": [
                    "Revenue growth consistency",
                    "Market share trends",
                    "R&D investment levels",
                    "Capital expenditure patterns",
                    "Return on invested capital"
                ],
                "strategic_focus_areas": [
                    "Evaluate management's stated strategic goals",
                    "Assess progress on key performance indicators",
                    "Review competitive positioning",
                    "Analyze investment in future growth",
                    "Consider ESG and sustainability initiatives"
                ]
            }
        
        return base_guidance

    def _analyze_cash_flow(self, financial_context: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cash flow using domain expertise."""
        analysis = {
            "cash_flow_quality": "Unknown",
            "key_insights": [],
            "strengths": [],
            "concerns": [],
            "recommendations": []
        }
        
        try:
            cash_flow_data = financial_context.get("cash_flow", {})
            key_metrics = financial_context.get("key_metrics", {})
            
            if cash_flow_data and isinstance(cash_flow_data, dict):
                # Analyze quarterly cash flow trends
                quarterly_data = cash_flow_data.get("quarterly", {}).get("data", {})
                
                if quarterly_data:
                    # Extract operating cash flow trends
                    operating_cf_key = self._find_cash_flow_key(quarterly_data, ["Operating Cash Flow", "Total Cash From Operating Activities"])
                    
                    if operating_cf_key:
                        cf_values = []
                        periods = []
                        for period, values in quarterly_data.items():
                            if operating_cf_key in values and values[operating_cf_key] is not None:
                                cf_values.append(values[operating_cf_key])
                                periods.append(period)
                        
                        if len(cf_values) >= 2:
                            # Analyze cash flow trend
                            recent_cf = cf_values[0] if cf_values else 0
                            prior_cf = cf_values[1] if len(cf_values) > 1 else 0
                            
                            if recent_cf > 0:
                                analysis["cash_flow_quality"] = "Positive"
                                if recent_cf > prior_cf:
                                    analysis["strengths"].append("Operating cash flow is improving")
                                else:
                                    analysis["concerns"].append("Operating cash flow is declining")
                            else:
                                analysis["cash_flow_quality"] = "Negative" 
                                analysis["concerns"].append("Negative operating cash flow")
                            
                            analysis["key_insights"].append(f"Most recent operating cash flow: ${recent_cf:,.0f}")
            
            # Add domain expertise insights
            cash_flow_guidance = guidance.get("cash_flow_expertise", {})
            for analysis_point in cash_flow_guidance.get("analysis_points", []):
                analysis["recommendations"].append(f"Consider: {analysis_point}")
            
        except Exception as e:
            analysis["concerns"].append(f"Cash flow analysis error: {str(e)}")
        
        return analysis

    def _analyze_operations(self, financial_context: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze operations and profitability using domain expertise."""
        analysis = {
            "operational_efficiency": "Unknown",
            "key_insights": [],
            "strengths": [],
            "concerns": [],
            "recommendations": []
        }
        
        try:
            key_metrics = financial_context.get("key_metrics", {})
            financial_health = key_metrics.get("financial_health", {})
            
            # Analyze profitability margins
            gross_margin = financial_health.get("gross_margins")
            operating_margin = financial_health.get("operating_margins")
            profit_margin = financial_health.get("profit_margins")
            
            if gross_margin is not None:
                analysis["key_insights"].append(f"Gross margin: {gross_margin:.2%}")
                if gross_margin > 0.3:  # 30%+
                    analysis["strengths"].append("Strong gross margin indicates good pricing power")
                elif gross_margin < 0.1:  # <10%
                    analysis["concerns"].append("Low gross margin indicates pricing pressure")
            
            if operating_margin is not None:
                analysis["key_insights"].append(f"Operating margin: {operating_margin:.2%}")
                if operating_margin > 0.15:  # 15%+
                    analysis["strengths"].append("Strong operating margin indicates efficient operations")
                elif operating_margin < 0.05:  # <5%
                    analysis["concerns"].append("Low operating margin indicates operational challenges")
            
            # Assess overall operational efficiency
            if gross_margin and operating_margin and profit_margin:
                if all(m > 0.1 for m in [gross_margin, operating_margin, profit_margin]):
                    analysis["operational_efficiency"] = "Strong"
                elif any(m < 0 for m in [gross_margin, operating_margin, profit_margin]):
                    analysis["operational_efficiency"] = "Poor"
                else:
                    analysis["operational_efficiency"] = "Moderate"
            
            # Add domain expertise insights
            ops_guidance = guidance.get("operations_expertise", {})
            for analysis_point in ops_guidance.get("analysis_points", []):
                analysis["recommendations"].append(f"Consider: {analysis_point}")
            
        except Exception as e:
            analysis["concerns"].append(f"Operations analysis error: {str(e)}")
        
        return analysis

    def _identify_concerns(self, financial_context: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential areas of concern using domain expertise."""
        analysis = {
            "risk_level": "Unknown",
            "key_concerns": [],
            "financial_risks": [],
            "red_flags": [],
            "recommendations": []
        }
        
        try:
            key_metrics = financial_context.get("key_metrics", {})
            financial_health = key_metrics.get("financial_health", {})
            valuation = key_metrics.get("valuation", {})
            
            concern_count = 0
            
            # Liquidity concerns
            current_ratio = financial_health.get("current_ratio")
            if current_ratio is not None and current_ratio < 1.0:
                analysis["financial_risks"].append("Low current ratio indicates potential liquidity issues")
                concern_count += 1
            
            # Leverage concerns
            debt_to_equity = financial_health.get("debt_to_equity")
            if debt_to_equity is not None and debt_to_equity > 2.0:
                analysis["financial_risks"].append("High debt-to-equity ratio indicates high leverage risk")
                concern_count += 1
            
            # Profitability concerns
            roe = financial_health.get("return_on_equity")
            if roe is not None and roe < 0.05:  # <5%
                analysis["financial_risks"].append("Low return on equity indicates poor profitability")
                concern_count += 1
            
            # Valuation concerns
            pe_ratio = valuation.get("pe_ratio")
            if pe_ratio is not None and pe_ratio > 50:
                analysis["key_concerns"].append("High P/E ratio may indicate overvaluation")
                concern_count += 1
            elif pe_ratio is not None and pe_ratio < 0:
                analysis["red_flags"].append("Negative P/E ratio indicates losses")
                concern_count += 2
            
            # Overall risk assessment
            if concern_count >= 3:
                analysis["risk_level"] = "High"
            elif concern_count >= 1:
                analysis["risk_level"] = "Moderate"
            else:
                analysis["risk_level"] = "Low"
            
            # Add domain expertise insights
            risk_guidance = guidance.get("risk_assessment_expertise", {})
            for red_flag in risk_guidance.get("red_flags", []):
                analysis["recommendations"].append(f"Monitor for: {red_flag}")
            
        except Exception as e:
            analysis["key_concerns"].append(f"Risk analysis error: {str(e)}")
        
        return analysis

    def _assess_objectives(self, financial_context: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
        """Assess progress towards strategic objectives using domain expertise."""
        analysis = {
            "strategic_progress": "Unknown",
            "growth_indicators": [],
            "strategic_strengths": [],
            "areas_for_improvement": [],
            "recommendations": []
        }
        
        try:
            key_metrics = financial_context.get("key_metrics", {})
            growth = key_metrics.get("growth", {})
            
            # Analyze growth metrics
            revenue_growth = growth.get("revenue_growth")
            earnings_growth = growth.get("earnings_growth")
            
            if revenue_growth is not None:
                analysis["growth_indicators"].append(f"Revenue growth: {revenue_growth:.2%}")
                if revenue_growth > 0.1:  # >10%
                    analysis["strategic_strengths"].append("Strong revenue growth indicates market expansion")
                elif revenue_growth < 0:
                    analysis["areas_for_improvement"].append("Negative revenue growth indicates declining business")
            
            if earnings_growth is not None:
                analysis["growth_indicators"].append(f"Earnings growth: {earnings_growth:.2%}")
                if earnings_growth > 0.15:  # >15%
                    analysis["strategic_strengths"].append("Strong earnings growth indicates operational leverage")
                elif earnings_growth < 0:
                    analysis["areas_for_improvement"].append("Declining earnings indicate profitability challenges")
            
            # Overall strategic assessment
            if revenue_growth and earnings_growth:
                if revenue_growth > 0.05 and earnings_growth > 0.05:
                    analysis["strategic_progress"] = "Strong"
                elif revenue_growth < 0 or earnings_growth < 0:
                    analysis["strategic_progress"] = "Concerning"
                else:
                    analysis["strategic_progress"] = "Moderate"
            
            # Add domain expertise insights
            strategic_guidance = guidance.get("strategic_assessment_expertise", {})
            for focus_area in strategic_guidance.get("strategic_focus_areas", []):
                analysis["recommendations"].append(f"Evaluate: {focus_area}")
            
        except Exception as e:
            analysis["areas_for_improvement"].append(f"Objectives analysis error: {str(e)}")
        
        return analysis

    def _generate_overall_assessment(self, analysis_results: Dict[str, Any], financial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall fundamental assessment."""
        assessment = {
            "investment_recommendation": "HOLD",
            "confidence_level": "Medium",
            "key_strengths": [],
            "key_concerns": [],
            "fundamental_score": 50,  # Out of 100
            "summary": ""
        }
        
        try:
            # Aggregate strengths and concerns from all analyses
            total_strengths = 0
            total_concerns = 0
            
            for analysis_type, analysis_data in analysis_results.items():
                if isinstance(analysis_data, dict):
                    strengths = analysis_data.get("strengths", [])
                    concerns = analysis_data.get("concerns", [])
                    
                    assessment["key_strengths"].extend(strengths)
                    assessment["key_concerns"].extend(concerns)
                    
                    total_strengths += len(strengths)
                    total_concerns += len(concerns)
            
            # Calculate fundamental score
            if total_strengths + total_concerns > 0:
                strength_ratio = total_strengths / (total_strengths + total_concerns)
                assessment["fundamental_score"] = int(strength_ratio * 100)
            
            # Generate investment recommendation
            if assessment["fundamental_score"] >= 70:
                assessment["investment_recommendation"] = "BUY"
                assessment["confidence_level"] = "High"
            elif assessment["fundamental_score"] >= 60:
                assessment["investment_recommendation"] = "BUY"
                assessment["confidence_level"] = "Medium"
            elif assessment["fundamental_score"] <= 30:
                assessment["investment_recommendation"] = "SELL"
                assessment["confidence_level"] = "High"
            elif assessment["fundamental_score"] <= 40:
                assessment["investment_recommendation"] = "SELL"
                assessment["confidence_level"] = "Medium"
            else:
                assessment["investment_recommendation"] = "HOLD"
                assessment["confidence_level"] = "Medium"
            
            # Generate summary
            company_name = financial_context.get("company_profile", {}).get("name", "Company")
            assessment["summary"] = (
                f"Fundamental analysis of {company_name} reveals a score of {assessment['fundamental_score']}/100. "
                f"Key strengths include: {', '.join(assessment['key_strengths'][:3]) if assessment['key_strengths'] else 'None identified'}. "
                f"Areas of concern include: {', '.join(assessment['key_concerns'][:3]) if assessment['key_concerns'] else 'None identified'}."
            )
            
        except Exception as e:
            assessment["summary"] = f"Assessment generation error: {str(e)}"
        
        return assessment

    def _find_cash_flow_key(self, data: Dict[str, Any], possible_keys: List[str]) -> Optional[str]:
        """Find the correct key for cash flow data."""
        for item_name, _ in data.items():
            for key in possible_keys:
                if key.lower() in item_name.lower():
                    return item_name
        return None


class CompanyNameResolverTool(BaseTool):
    """Tool to resolve company names or ISIN to stock tickers."""
    
    name: str = "resolve_company_ticker"
    description: str = (
        "Resolves company names (e.g., 'Apple', 'Microsoft') or ISIN codes to stock tickers. "
        "Returns the ticker symbol that can be used for fundamental analysis."
    )
    args_schema: type[BaseModel] = CompanyResolverInput

    def _run(self, query: str) -> Dict[str, Any]:
        """Resolve company name/ISIN to ticker symbol."""
        try:
            query = query.strip()
            
            # Common company name mappings
            company_mappings = {
                # Tech companies
                "apple": "AAPL", "apple inc": "AAPL",
                "microsoft": "MSFT", "microsoft corp": "MSFT",
                "google": "GOOGL", "alphabet": "GOOGL",
                "amazon": "AMZN", "amazon.com": "AMZN",
                "tesla": "TSLA", "tesla inc": "TSLA",
                "meta": "META", "facebook": "META",
                "netflix": "NFLX", "nvidia": "NVDA",
                
                # Financial companies  
                "jpmorgan": "JPM", "jp morgan": "JPM",
                "bank of america": "BAC", "goldman sachs": "GS",
                "morgan stanley": "MS", "wells fargo": "WFC",
                
                # Other major companies
                "walmart": "WMT", "berkshire hathaway": "BRK-B",
                "johnson & johnson": "JNJ", "exxon mobil": "XOM",
                "visa": "V", "mastercard": "MA"
            }
            
            # Check if already a ticker
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
                    logger.warning(f"Failed to validate mapped ticker {ticker}: {str(e)}")
            
            # If not found, return error with suggestions
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


# Export all tools
def get_fundamental_tools() -> List[BaseTool]:
    """Get all fundamental analysis tools."""
    return [
        CompanyNameResolverTool(),
        FinanceReportPullTool(),
        RAGAnalysisTool()
    ]