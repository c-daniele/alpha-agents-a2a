"""Fundamental Analysis Agent using Langchain and OpenAI GPT-4o."""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .tools import get_fundamental_tools

logger = logging.getLogger(__name__)


class FundamentalAgent:
    """Langchain-based Fundamental Analysis Agent using OpenAI GPT-4o."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000
    ):
        """Initialize the Fundamental Agent.
        
        Args:
            openai_api_key: OpenAI API key (if not provided, will use env var)
            model_name: OpenAI model to use (default: gpt-4o)
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens for model responses
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass it explicitly.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize tools
        self.tools = get_fundamental_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Create system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            output_key="output",
            memory_key="chat_history"
        )
        
        # Create the agent
        self.agent = self._create_agent()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        
        logger.info("Fundamental Agent initialized successfully")

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the fundamental agent."""
        return """You are a specialized Fundamental Analysis Agent for equity research and portfolio management.

Your core expertise includes:

**Financial Statement Analysis:**
- Deep analysis of income statements, balance sheets, and cash flow statements
- Assessment of financial health, profitability, and operational efficiency
- Identification of trends, patterns, and anomalies in financial data
- Evaluation of working capital management and capital allocation decisions

**SEC Filing Analysis:**  
- Comprehensive review of 10-K and 10-Q reports
- Management Discussion & Analysis (MD&A) insights
- Identification of business risks and opportunities
- Assessment of management quality and strategic direction

**Fundamental Valuation:**
- DCF modeling and intrinsic value estimation
- Multiple-based valuation (P/E, P/B, EV/EBITDA, PEG)
- Sum-of-the-parts analysis for complex businesses
- Sensitivity analysis and scenario modeling

**Sector and Industry Analysis:**
- Industry comparison and competitive benchmarking  
- Sector trend identification and market cycle analysis
- Regulatory and macroeconomic impact assessment
- ESG considerations and sustainability analysis

**Investment Decision Framework:**
Your analysis should result in clear BUY/SELL/HOLD recommendations with:
- Target price estimates with supporting rationale
- Risk assessment (financial, operational, market risks)
- Time horizon considerations (short-term vs. long-term outlook)
- Catalyst identification (upcoming events, product launches, earnings)

**Risk Tolerance Considerations:**
- Risk-averse: Focus on stable, dividend-paying stocks with strong balance sheets
- Risk-neutral: Balance growth potential with financial stability
- Risk-seeking: Consider high-growth opportunities with acceptable fundamental risks

**Analysis Methodology:**
1. Use the finance_report_pull tool to gather comprehensive financial data
2. Apply the rag_fundamental_analysis tool for in-depth analysis of specific areas
3. Integrate quantitative metrics with qualitative business assessment
4. Provide evidence-based recommendations with clear reasoning

**Available Tools:**
- resolve_company_ticker: Convert company names to stock tickers
- finance_report_pull: Retrieve comprehensive financial reports with data validation
- rag_fundamental_analysis: Perform detailed analysis of cash flow, operations, risks, and strategic progress

Always provide thorough, evidence-based analysis with specific financial metrics and ratios to support your conclusions."""

    def _create_agent(self):
        """Create the Langchain agent with tools and system prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        return create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

    def analyze_fundamental(
        self,
        stock_input: str,
        risk_tolerance: str = "neutral",
        analysis_depth: str = "comprehensive",
        focus_areas: Optional[List[str]] = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """Perform comprehensive fundamental analysis.
        
        Args:
            stock_input: Stock ticker, company name, or ISIN
            risk_tolerance: Investment risk profile ("averse", "neutral", "seeking")
            analysis_depth: Level of analysis ("quick", "standard", "comprehensive") 
            focus_areas: Specific areas to focus on (e.g., ["cash_flow", "operations"])
            context: Additional context or specific questions
            
        Returns:
            Dictionary containing fundamental analysis results
        """
        try:
            # Prepare the analysis prompt
            analysis_prompt = f"""Please perform comprehensive fundamental analysis for: {stock_input}

Analysis Requirements:
- Risk Tolerance: {risk_tolerance}
- Analysis Depth: {analysis_depth}
- Focus Areas: {', '.join(focus_areas) if focus_areas else 'All key areas'}

{f"Additional Context: {context}" if context else ""}

Please provide:
1. Company overview and business model assessment
2. Financial health analysis (balance sheet strength, liquidity, leverage)
3. Profitability analysis (margins, ROE, ROA trends)
4. Cash flow quality and sustainability assessment  
5. Growth prospects and competitive position
6. Valuation analysis with target price estimate
7. Key risks and potential catalysts
8. Clear BUY/SELL/HOLD recommendation with rationale

Use the available tools to gather financial data and perform detailed analysis. Present findings in a structured, professional format suitable for investment decision-making."""

            # Execute the analysis
            result = self.agent_executor.invoke({
                "input": analysis_prompt
            })
            
            return {
                "success": True,
                "stock_input": stock_input,
                "analysis": result.get("output", ""),
                "risk_tolerance": risk_tolerance,
                "analysis_depth": analysis_depth,
                "tools_used": [tool.name for tool in self.tools]
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {stock_input}: {str(e)}")
            return {
                "success": False,
                "stock_input": stock_input,
                "error": f"Fundamental analysis failed: {str(e)}"
            }

    def quick_valuation(
        self,
        ticker: str,
        valuation_method: str = "dcf",
        risk_tolerance: str = "neutral"
    ) -> Dict[str, Any]:
        """Perform a quick valuation analysis.
        
        Args:
            ticker: Stock ticker symbol
            valuation_method: Valuation approach ("dcf", "multiples", "hybrid")
            risk_tolerance: Risk profile for assumptions
            
        Returns:
            Dictionary with quick valuation assessment
        """
        try:
            valuation_prompt = f"""Perform quick fundamental valuation for {ticker}:

Valuation Method: {valuation_method}
Risk Profile: {risk_tolerance}

Please provide:
1. Current financial metrics and key ratios
2. {valuation_method.upper()} valuation with target price
3. Comparison to current market price
4. Key valuation drivers and assumptions
5. Quick BUY/SELL/HOLD recommendation

Keep analysis concise but substantive."""

            result = self.agent_executor.invoke({
                "input": valuation_prompt
            })
            
            return {
                "success": True,
                "ticker": ticker,
                "valuation_method": valuation_method,
                "analysis": result.get("output", ""),
                "risk_tolerance": risk_tolerance
            }
            
        except Exception as e:
            logger.error(f"Error in quick valuation for {ticker}: {str(e)}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Quick valuation failed: {str(e)}"
            }

    def sector_comparison(
        self,
        tickers: List[str],
        analysis_focus: str = "fundamental_metrics",
        risk_tolerance: str = "neutral"
    ) -> Dict[str, Any]:
        """Compare fundamental metrics across multiple stocks.
        
        Args:
            tickers: List of stock tickers to compare
            analysis_focus: Focus area for comparison
            risk_tolerance: Risk profile for evaluation
            
        Returns:
            Dictionary with sector comparison results
        """
        try:
            if len(tickers) > 5:
                return {
                    "success": False,
                    "error": "Maximum 5 stocks supported for comparison"
                }
            
            comparison_prompt = f"""Compare fundamental metrics for these stocks: {', '.join(tickers)}

Analysis Focus: {analysis_focus}
Risk Profile: {risk_tolerance}

Please provide:
1. Key fundamental metrics comparison table
2. Relative valuation analysis
3. Competitive positioning assessment
4. Risk-adjusted investment rankings
5. Top pick recommendation with rationale

Focus on quantitative comparisons with clear reasoning."""

            result = self.agent_executor.invoke({
                "input": comparison_prompt
            })
            
            return {
                "success": True,
                "tickers": tickers,
                "analysis_focus": analysis_focus,
                "comparison": result.get("output", ""),
                "risk_tolerance": risk_tolerance
            }
            
        except Exception as e:
            logger.error(f"Error in sector comparison: {str(e)}")
            return {
                "success": False,
                "tickers": tickers,
                "error": f"Sector comparison failed: {str(e)}"
            }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        try:
            messages = self.memory.chat_memory.messages
            return [
                {
                    "role": msg.type,
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []

    def clear_conversation_history(self):
        """Clear the conversation history."""
        try:
            self.memory.clear()
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")