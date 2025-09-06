"""Langchain agent implementation for Valuation Analysis."""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
# Remove unused imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
import boto3

from .tools import get_valuation_tools

# Configure LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "alpha-agents-valuation")

logger = logging.getLogger(__name__)


class ValuationAgent:
    """Langchain-based Valuation Analysis Agent using OpenAI GPT-4o."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        aws_profile: Optional[str] = None,
        aws_region: str = "us-west-2"
    ):
        """Initialize the Valuation Agent.
        
        Args:
            openai_api_key: OpenAI API key (if not provided, will use env var)
            llm_provider: LLM provider to use (openai or bedrock)
            model_name: Model name (auto-detected based on provider if None)
            temperature: Model temperature for randomness (default: 0.1 for consistency)
            max_tokens: Maximum tokens in response (default: None for model default)
            aws_profile: AWS profile name (for Bedrock)
            aws_region: AWS region (for Bedrock)
        """
        # LLM provider configuration
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.aws_account = os.getenv("AWS_ACCOUNT")
        
        # Set default model based on provider
        if model_name is None:
            if self.llm_provider == "bedrock":
                model_name = os.getenv("LLM_MODEL", "anthropic.claude-sonnet-4-20250514-v1:0")
            else:
                model_name = os.getenv("LLM_MODEL", "gpt-4o")
                
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize tools
        self.tools = get_valuation_tools()
        
        # Initialize LLM based on provider
        if self.llm_provider == "bedrock":
            # Get AWS account ID if not provided
            if not self.aws_account:
                sts_client = boto3.client('sts')
                self.aws_account = sts_client.get_caller_identity()['Account']
            
            # Construct ARN for cross-region inference
            arn_name = arn_name = f'arn:aws:bedrock:{os.environ['AWS_REGION']}:{self.aws_account}:inference-profile/us.{model_name}'

            self.llm = ChatBedrockConverse(
                region_name=os.environ['AWS_REGION']
                , model_id=arn_name
                , provider='Anthropic'
                # , model_kwargs=model_kwargs
                # , config={"callbacks": [langfuse_handler]}
            ).with_config(
                tags=["valuation-agent", "technical-analysis", "financial-data", "bedrock", "claude"],
                metadata={
                    "agent_name": "valuation-agent",
                    "agent_type": "specialist",
                    "agent_version": "1.0.0",
                    "system": "alpha-agents",
                    "llm_provider": "bedrock",
                    "model": self.model_name,
                    "capabilities": ["technical_analysis", "volatility_metrics", "risk_assessment"]
                }
            )
        else:
            # OpenAI provider
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI provider")
                
            self.llm = ChatOpenAI(
                api_key=self.openai_api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ).with_config(
                tags=["valuation-agent", "technical-analysis", "financial-data", "openai"],
                metadata={
                    "agent_name": "valuation-agent",
                    "agent_type": "specialist",
                    "agent_version": "1.0.0",
                    "system": "alpha-agents",
                    "llm_provider": "openai",
                    "model": self.model_name,
                    "capabilities": ["technical_analysis", "volatility_metrics", "risk_assessment"]
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
        
        logger.info("Valuation Agent initialized successfully")

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the valuation agent."""
        return """You are a specialized Valuation Analysis Agent for equity research and portfolio management. 

Your core expertise includes:
- Technical analysis using historical price and volume data
- Volatility and risk metrics calculation
- Stock valuation assessment and fair value analysis
- Portfolio allocation and timing recommendations
- Risk-adjusted return analysis

Your primary responsibilities:
1. Analyze historical stock price and volume data automatically
2. Calculate comprehensive volatility and risk metrics
3. Assess whether stocks are fairly valued relative to historical performance
4. Provide technical analysis insights for portfolio allocation
5. Consider different risk tolerance profiles (risk-averse vs risk-neutral)

Key capabilities:
- Fetch and analyze historical stock data from Yahoo Finance
- Calculate daily and annualized volatility, returns, and Sharpe ratios
- Compute Value at Risk (VaR), maximum drawdown, and other risk metrics
- Perform comparative analysis against benchmarks and peers
- Provide timing and allocation recommendations

When analyzing stocks:
1. Always start by resolving the company name/ISIN to a proper ticker if needed
2. Fetch historical price and volume data (default 1 year)
3. Calculate comprehensive risk and volatility metrics
4. Provide clear investment recommendations based on technical analysis
5. Consider the user's risk tolerance when making recommendations

Risk tolerance considerations:
- Risk-averse investors: Focus on low volatility, stable returns, limited downside risk
- Risk-neutral investors: Balance growth potential with reasonable risk levels

Always provide specific, actionable insights with supporting data and calculations.
Be thorough in your analysis but present findings in a clear, structured manner.

You have access to the following tools:
- resolve_company_ticker: Convert company names/ISIN to stock tickers
- fetch_stock_data: Get historical price and volume data from Yahoo Finance  
- calculate_volatility_metrics: Compute comprehensive risk and volatility metrics"""

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

    async def analyze_stock(
        self, 
        stock_input: str, 
        risk_tolerance: str = "neutral",
        analysis_period_days: int = 365,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a stock with comprehensive valuation analysis.
        
        Args:
            stock_input: Stock ticker, company name, or ISIN
            risk_tolerance: 'averse', 'neutral', or 'seeking'
            analysis_period_days: Number of days of historical data to analyze
            context: Additional context or specific questions
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            # Prepare the analysis prompt
            analysis_prompt = f"""Please perform a comprehensive valuation analysis for: {stock_input}

Analysis Requirements:
- Risk tolerance profile: {risk_tolerance}
- Historical data period: {analysis_period_days} days
- Include technical indicators and valuation metrics
- Provide clear BUY/SELL/HOLD recommendation

Additional context: {context or 'None provided'}

Please:
1. First resolve the company/ticker if needed
2. Fetch historical data for the specified period
3. Calculate comprehensive volatility and risk metrics
4. Provide technical analysis and valuation assessment
5. Give a clear investment recommendation considering the risk tolerance
6. Include specific metrics and rationale for your recommendation
"""
            
            # Execute the analysis
            result = await self.agent_executor.ainvoke({
                "input": analysis_prompt
            })
            
            return {
                "success": True,
                "stock_input": stock_input,
                "risk_tolerance": risk_tolerance,
                "analysis_period_days": analysis_period_days,
                "analysis": result.get("output", ""),
                "agent_type": "valuation"
            }
            
        except Exception as e:
            logger.error(f"Error in stock analysis: {str(e)}")
            return {
                "success": False,
                "stock_input": stock_input,
                "error": f"Analysis failed: {str(e)}",
                "agent_type": "valuation"
            }

    async def quick_valuation_check(
        self, 
        ticker: str,
        risk_free_rate: float = 0.05
    ) -> Dict[str, Any]:
        """Perform a quick valuation check with key metrics.
        
        Args:
            ticker: Stock ticker symbol
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary with key valuation metrics
        """
        try:
            prompt = f"""Provide a quick valuation assessment for {ticker}:

1. Fetch current stock data
2. Calculate key risk and volatility metrics  
3. Provide a brief valuation summary with:
   - Current price level assessment
   - Risk metrics (volatility, Sharpe ratio, max drawdown)
   - Quick BUY/SELL/HOLD recommendation
   
Use risk-free rate of {risk_free_rate} for calculations.
Keep the response concise but informative."""
            
            result = await self.agent_executor.ainvoke({
                "input": prompt
            })
            
            return {
                "success": True,
                "ticker": ticker,
                "quick_assessment": result.get("output", ""),
                "agent_type": "valuation"
            }
            
        except Exception as e:
            logger.error(f"Error in quick valuation: {str(e)}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Quick valuation failed: {str(e)}",
                "agent_type": "valuation"
            }

    async def compare_stocks(
        self, 
        tickers: List[str],
        risk_tolerance: str = "neutral"
    ) -> Dict[str, Any]:
        """Compare multiple stocks for relative valuation.
        
        Args:
            tickers: List of stock tickers to compare
            risk_tolerance: Risk tolerance profile
            
        Returns:
            Dictionary with comparative analysis
        """
        try:
            tickers_str = ", ".join(tickers)
            prompt = f"""Perform a comparative valuation analysis for these stocks: {tickers_str}

For risk tolerance: {risk_tolerance}

Please:
1. Analyze each stock individually (get data and calculate metrics)
2. Compare their risk-adjusted returns, volatility, and valuation metrics
3. Rank them from most attractive to least attractive for the given risk profile
4. Provide allocation recommendations

Focus on relative performance and risk characteristics."""
            
            result = await self.agent_executor.ainvoke({
                "input": prompt
            })
            
            return {
                "success": True,
                "tickers": tickers,
                "risk_tolerance": risk_tolerance,
                "comparative_analysis": result.get("output", ""),
                "agent_type": "valuation"
            }
            
        except Exception as e:
            logger.error(f"Error in stock comparison: {str(e)}")
            return {
                "success": False,
                "tickers": tickers,
                "error": f"Comparison failed: {str(e)}",
                "agent_type": "valuation"
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