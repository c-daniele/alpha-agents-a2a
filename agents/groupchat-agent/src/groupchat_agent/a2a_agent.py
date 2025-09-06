"""A2A-compliant GroupChat Agent for coordinating multi-agent analysis."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import traceback
import boto3
from langchain_aws import ChatBedrockConverse

# Configure LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "alpha-agents-groupchat")

from .registry_service import AgentRegistryService

logger = logging.getLogger(__name__)


class A2AGroupChatAgent:
    """
    A2A-compliant GroupChat Agent that uses LLM to intelligently coordinate 
    with specialist agents based on user requests.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        registry_url: Optional[str] = None,
        valuation_agent_name: str = "valuation",
        sentiment_agent_name: str = "sentiment",
        fundamental_agent_name: str = "fundamental",
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        aws_profile: Optional[str] = None,
        aws_region: str = "us-west-2"
    ):
        """Initialize the A2A GroupChat Agent."""
        
        # LLM provider configuration
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.aws_account = os.getenv("AWS_ACCOUNT")  # Will be retrieved if not set
        
        # Set default model based on provider
        if model_name is None:
            if self.llm_provider == "bedrock":
                model_name = os.getenv("LLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            else:
                model_name = os.getenv("LLM_MODEL", "gpt-4o")
        
        self.registry_url = registry_url or os.getenv("AGENT_REGISTRY_URL")
        if not self.registry_url:
            raise ValueError("Agent Registry URL is required")
            
        # Agent names for registry lookup
        self.valuation_agent_name = valuation_agent_name
        self.sentiment_agent_name = sentiment_agent_name
        self.fundamental_agent_name = fundamental_agent_name
        
        # Initialize services
        self.registry_service = AgentRegistryService(self.registry_url)

        print(f"Initializing A2A GroupChat Agent with LLM provider: {self.llm_provider}, model: {model_name}")
        
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
                tags=["groupchat-agent", "coordinator", "multi-agent", "bedrock", "claude"],
                metadata={
                    "agent_name": "groupchat-agent",
                    "agent_type": "coordinator",
                    "agent_version": "1.0.0",
                    "system": "alpha-agents",
                    "llm_provider": "bedrock",
                    "model": model_name
                }
            )
        else:
            # OpenAI provider
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI provider")
                
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.openai_api_key,
                temperature=0.1
            ).with_config(
                tags=["groupchat-agent", "coordinator", "multi-agent", "openai"],
                metadata={
                    "agent_name": "groupchat-agent",
                    "agent_type": "coordinator",
                    "agent_version": "1.0.0",
                    "system": "alpha-agents",
                    "llm_provider": "openai",
                    "model": model_name
                }
            )
        
        # Cache for agent URLs
        self._agent_urls = {}
        self._agents_fetched = False
        
        logger.info("A2A GroupChat Agent initialized successfully")

    async def _ensure_agent_urls(self) -> bool:
        """Ensure specialist agent URLs are fetched from registry."""
        if self._agents_fetched and self._agent_urls:
            return True
            
        try:
            agent_urls = await self.registry_service.get_specialist_agents(
                valuation_name=self.valuation_agent_name,
                sentiment_name=self.sentiment_agent_name,
                fundamental_name=self.fundamental_agent_name
            )
            
            missing_agents = [name for name, url in agent_urls.items() if url is None]
            if missing_agents:
                logger.error(f"Missing agents in registry: {missing_agents}")
                return False
                
            self._agent_urls = agent_urls
            self._agents_fetched = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch agent URLs: {e}")
            return False

    def _determine_required_agents(self, user_message: str) -> List[str]:
        """Use LLM to determine which specialist agents are needed."""
        
        system_prompt = """You are a routing coordinator for financial analysis agents. 
Analyze the user's request and determine which specialist agents should be consulted:

AVAILABLE AGENTS:
- valuation: Technical analysis, price charts, volatility, quantitative metrics
- sentiment: News analysis, market sentiment, social media trends  
- fundamental: Financial reports, earnings, company fundamentals, SEC filings

ROUTING RULES:
1. For specific analysis types (e.g. "sentiment analysis"), use only that agent
2. For comprehensive analysis, use multiple relevant agents
3. For general investment questions, use all three agents
4. Consider the scope and complexity of the request

Respond with ONLY a JSON list of agent names needed, like: ["valuation"] or ["valuation", "sentiment", "fundamental"]"""

        try:
            # Log the complete routing prompt for detailed analysis
            full_prompt = f"SYSTEM: {system_prompt}\n\nHUMAN: User request: {user_message}"
            logger.info(f"🤖 LLM ROUTING PROMPT:\n{full_prompt}")
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User request: {user_message}")
            ])
            
            # Log the complete LLM response
            logger.info(f"🤖 LLM ROUTING RESPONSE: {response.content}")
            
            # Parse LLM response to get agent list
            agent_list_text = response.content.strip()
            
            # Try to extract JSON from the response
            if agent_list_text.startswith('[') and agent_list_text.endswith(']'):
                agents = json.loads(agent_list_text)
            else:
                # Fallback: look for keywords
                lower_message = user_message.lower()
                agents = []
                
                if any(word in lower_message for word in ['sentiment', 'news', 'social', 'opinion']):
                    agents.append('sentiment')
                if any(word in lower_message for word in ['technical', 'chart', 'price', 'volatility', 'valuation']):
                    agents.append('valuation')  
                if any(word in lower_message for word in ['fundamental', 'earnings', 'financial', 'report', '10-k']):
                    agents.append('fundamental')
                
                # Default to all agents for comprehensive requests
                if not agents or any(word in lower_message for word in ['comprehensive', 'complete', 'full', 'overall', 'analysis', 'invest']):
                    agents = ['valuation', 'sentiment', 'fundamental']
            
            logger.info(f"Determined required agents: {agents}")
            return agents
            
        except Exception as e:
            logger.error(f"Error determining required agents: {e}\n{traceback.format_exc()}")
            # Default to all agents on error
            return ['valuation', 'sentiment', 'fundamental']

    async def _send_message_to_agent(
        self, 
        agent_name: str, 
        agent_url: str,
        message: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send A2A message/send request to a specialist agent."""
        
        try:
            # Construct A2A-compliant request
            request_payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": message,
                    "metadata": metadata or {}
                },
                "id": str(uuid.uuid4())
            }
            
            logger.info(f"Sending A2A message to {agent_name} at {agent_url}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60.0)) as session:
                
                timeout = aiohttp.ClientTimeout(total=300.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        agent_url,
                        json=request_payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        result = await response.json()

                    if "error" in result and result["error"] is not None:
                        logger.error(f"A2A error from {agent_name}: {result['error']}")
                        return {
                            "agent": agent_name,
                            "success": False,
                            "error": result["error"]["message"],
                            "analysis": f"Error from {agent_name}: {result['error']['message']}"
                        }
                    
                    # Extract message content from A2A response
                    a2a_result = result.get("result", {})
                    
                    # Handle both Task and Message responses
                    if isinstance(a2a_result, dict):
                        if a2a_result.get("id") and a2a_result.get("status"):
                            # Task response - wait for completion and get result
                            analysis = await self._handle_task_response(agent_name, agent_url, a2a_result)
                        elif a2a_result.get("kind") == "message":
                            # Direct message response
                            parts = a2a_result.get("parts", [])
                            analysis = " ".join([p.get("text", "") for p in parts if p.get("kind") == "text"])
                        else:
                            analysis = str(a2a_result)
                    else:
                        analysis = str(a2a_result)
                    
                    return {
                        "agent": agent_name,
                        "success": True,
                        "analysis": analysis,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "raw_response": a2a_result
                    }
                    
        except Exception as e:
            logger.error(f"Error communicating with {agent_name}: {e}")
            stack_trace = traceback.format_exc()
            logger.error(f"{e}\nStacktrace:\n{stack_trace}")
            
            return {
                "agent": agent_name,
                "success": False,
                "error": str(e),
                "analysis": f"Communication error with {agent_name}: {str(e)}"
            }

    async def _handle_task_response(
        self, 
        agent_name: str, 
        agent_url: str, 
        task_response: Dict[str, Any],
        max_wait_time: int = 120,
        poll_interval: int = 2
    ) -> str:
        """Handle task-based responses by polling for completion."""
        
        task_id = task_response.get("id")
        if not task_id:
            return f"Invalid task response from {agent_name}: no task ID"
        
        logger.info(f"Waiting for task {task_id} from {agent_name} to complete...")
        
        # Poll for task completion
        for _ in range(max_wait_time // poll_interval):
            try:
                # Create tasks/get request
                task_get_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/get", 
                    "params": {"id": task_id},
                    "id": str(uuid.uuid4())
                }
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
                    async with session.post(
                        agent_url,
                        json=task_get_payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status != 200:
                            logger.warning(f"Task polling failed for {agent_name}: HTTP {response.status}")
                            await asyncio.sleep(poll_interval)
                            continue
                            
                        task_result = await response.json()

                        if "error" in task_result and task_result["error"] is not None:
                            logger.error(f"Task get error from {agent_name}: {task_result['error']}")
                            return f"Task polling error from {agent_name}: {task_result['error']['message']}"
                        
                        task_data = task_result.get("result", {})
                        task_status = task_data.get("status", {})
                        state = task_status.get("state", "unknown")
                        
                        logger.info(f"Task {task_id} state: {state}")
                        
                        if state == "completed":
                            # Extract the final message
                            message = task_status.get("message")
                            if message and isinstance(message, dict):
                                parts = message.get("parts", [])
                                analysis = " ".join([p.get("text", "") for p in parts if p.get("kind") == "text"])
                                if analysis.strip():
                                    logger.info(f"Task {task_id} completed successfully")
                                    return analysis
                            
                            return f"Task completed but no analysis content available from {agent_name}"
                            
                        elif state == "failed":
                            error_msg = task_status.get("error", "Unknown task error")
                            logger.error(f"Task {task_id} failed: {error_msg}")
                            return f"Task failed from {agent_name}: {error_msg}"
                        
                        elif state in ["pending", "running"]:
                            # Task still processing, continue polling
                            logger.debug(f"Task {task_id} still {state}, waiting...")
                            await asyncio.sleep(poll_interval)
                            continue
                        
                        else:
                            logger.warning(f"Unknown task state: {state}")
                            await asyncio.sleep(poll_interval)
                            continue
                            
            except Exception as e:
                logger.warning(f"Error polling task {task_id}: {e}")
                logger.warning(f"Stacktrace:\n{traceback.format_exc()}")
                await asyncio.sleep(poll_interval)
                continue
        
        # Timeout reached
        logger.error(f"Task {task_id} from {agent_name} timed out after {max_wait_time}s")
        return f"Task timeout from {agent_name}: analysis took longer than {max_wait_time} seconds"

    async def _coordinate_agents(
        self,
        user_message: str,
        context_id: str,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Coordinate with required specialist agents based on user request."""
        
        # Ensure agent URLs are available
        if not await self._ensure_agent_urls():
            raise ValueError("Failed to fetch specialist agent URLs from registry")
        
        # Determine which agents are needed
        required_agents = self._determine_required_agents(user_message)
        
        # Check if conflicting analysis is expected - trigger debate mechanism
        if len(required_agents) > 1 and self._requires_debate_analysis(user_message):
            logger.info("Multi-agent analysis detected - initiating structured debate")
            return await self._conduct_structured_debate(user_message, context_id, task_id, required_agents, metadata)
        else:
            # Single agent or simple coordination
            return await self._simple_coordination(user_message, context_id, task_id, required_agents, metadata)
    
    def _requires_debate_analysis(self, user_message: str) -> bool:
        """Use LLM to intelligently determine if the user request requires structured debate between agents."""
        
        system_prompt = """You are a debate necessity classifier for a multi-agent financial analysis system. Your task is to determine if a user query requires collaborative decision-making through structured debate between specialist agents (fundamental, sentiment, and valuation experts).

STRUCTURED DEBATE IS NEEDED when:
1. The query asks for investment decisions or recommendations (buy/sell/hold advice)
2. The query requires weighing conflicting factors or multiple perspectives
3. The query asks for comprehensive analysis that would benefit from specialist disagreement/consensus
4. The query involves risk assessment or strategic financial decisions
5. The query asks for opinions, advice, or evaluations that could have multiple valid perspectives
6. The query involves portfolio management decisions
7. The query asks to "analyze," "evaluate," "assess," or "recommend" regarding investments

STRUCTURED DEBATE IS NOT NEEDED when:
1. The query asks for simple factual information (stock prices, company data)
2. The query is about specific technical analysis only
3. The query is about news sentiment only  
4. The query is about fundamental data only
5. The query asks for definitions or explanations
6. The query is a simple informational request that doesn't require decision-making

The query can be in ANY language. Focus on the intent and meaning, not specific keywords.

Respond with ONLY "YES" if structured debate is needed, or "NO" if it's not needed. No explanation."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {user_message}")
            ])
            
            decision = response.content.strip().upper()
            requires_debate = decision == "YES"
            
            # Log the LLM's decision for transparency
            logger.info(f"🤔 LLM DEBATE DECISION: {decision} for query: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")
            
            return requires_debate
            
        except Exception as e:
            logger.error(f"Error in LLM debate decision: {e}")
            # Fallback: if multiple agents are required, likely needs debate
            required_agents = self._determine_required_agents(user_message)
            fallback_decision = len(required_agents) > 1
            logger.info(f"Using fallback debate decision: {fallback_decision} (multiple agents required: {len(required_agents)})")
            return fallback_decision

    def _enhance_metadata_with_risk_tolerance(
        self, 
        user_message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract or infer risk tolerance and add it to metadata."""
        
        # Start with existing metadata or empty dict
        enhanced_metadata = metadata.copy() if metadata else {}
        
        # If risk_tolerance already provided in metadata, use it
        if "risk_tolerance" in enhanced_metadata:
            logger.info(f"Using provided risk_tolerance: {enhanced_metadata['risk_tolerance']}")
            return enhanced_metadata
        
        # Otherwise, try to infer risk tolerance from user message using LLM
        risk_tolerance = self._infer_risk_tolerance_from_message(user_message)
        enhanced_metadata["risk_tolerance"] = risk_tolerance
        
        logger.info(f"Inferred risk_tolerance: {risk_tolerance} from user message")
        return enhanced_metadata

    def _infer_risk_tolerance_from_message(self, user_message: str) -> str:
        """Use LLM to infer risk tolerance from user message content."""
        
        system_prompt = """You are a financial risk tolerance classifier. Analyze the user's message to determine their risk tolerance profile based on the language, intent, and context.

RISK TOLERANCE LEVELS:
- "averse": Conservative investors who prioritize capital preservation, stable returns, dividend-paying stocks, low volatility, safety-first approach
- "neutral": Balanced investors who seek moderate growth with reasonable risk, diversified portfolios, standard market exposure  
- "seeking": Aggressive investors who pursue high returns, accept high volatility, growth stocks, speculative investments

CLASSIFICATION GUIDELINES:
1. Look for explicit risk preferences ("conservative", "aggressive", "high growth", "safe", "stable")
2. Consider investment timeframe mentions ("retirement", "long-term", "quick gains")
3. Analyze vocabulary choices and tone (cautious vs confident vs speculative)
4. Consider asset mentions (bonds/dividends = averse, growth stocks = seeking)
5. Default to "neutral" if unclear or no strong risk indicators

The message can be in any language. Focus on the intent and risk signals.

Respond with ONLY one word: "averse", "neutral", or "seeking"."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User message: {user_message}")
            ])
            
            risk_level = response.content.strip().lower()
            
            # Validate response
            if risk_level in ["averse", "neutral", "seeking"]:
                return risk_level
            else:
                logger.warning(f"LLM returned invalid risk_tolerance '{risk_level}', defaulting to 'neutral'")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error inferring risk tolerance from message: {e}")
            return "neutral"  # Safe default
    
    async def _simple_coordination(
        self,
        user_message: str,
        context_id: str,
        task_id: str,
        required_agents: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Simple concurrent coordination without debate mechanism."""
        
        # Construct A2A message
        message = {
            "kind": "message",
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": user_message
                }
            ],
            "contextId": context_id,
            "taskId": task_id
        }
        
        # Send requests to required agents concurrently
        tasks = []
        for agent_name in required_agents:
            agent_url = self._agent_urls.get(agent_name)
            if agent_url:
                tasks.append(
                    self._send_message_to_agent(agent_name, agent_url, message, metadata)
                )
        
        if not tasks:
            raise ValueError("No valid specialist agents found for coordination")
        
        # Execute requests concurrently  
        agent_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        results = []
        for i, response in enumerate(agent_responses):
            if isinstance(response, Exception):
                agent_name = required_agents[i] if i < len(required_agents) else "unknown"
                results.append({
                    "agent": agent_name,
                    "success": False,
                    "error": str(response),
                    "analysis": f"Exception from {agent_name}: {str(response)}"
                })
            else:
                results.append(response)
        
        return results

    async def _conduct_structured_debate(
        self,
        user_message: str,
        context_id: str,
        task_id: str,
        required_agents: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        max_rounds: int = 5,
        min_turns_per_agent: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Conduct structured debate with strict turn-taking as per AlphaAgents paper.
        Each agent speaks at least twice, with round-robin turn taking.
        """
        logger.info(f"Starting structured debate with agents: {required_agents}")
        
        # Track agent participation
        agent_turns = {agent: 0 for agent in required_agents}
        debate_history = []
        current_positions = {}
        
        # Initial round - each agent provides their initial analysis
        logger.info("=== INITIAL ANALYSIS ROUND ===")
        initial_message = {
            "kind": "message",
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": f"{user_message}\n\nProvide your initial analysis and recommendation. This is the first round of a structured debate with other specialist agents."
                }
            ],
            "contextId": context_id,
            "taskId": task_id
        }
        
        # Get initial positions from each agent sequentially (turn-based)
        for agent_name in required_agents:
            agent_url = self._agent_urls.get(agent_name)
            if not agent_url:
                continue
                
            logger.info(f"Getting initial analysis from {agent_name} agent (risk_tolerance: {metadata.get('risk_tolerance', 'unknown')})")
            response = await self._send_message_to_agent(agent_name, agent_url, initial_message, metadata)
            
            if response.get("success", False):
                current_positions[agent_name] = response.get("analysis", "")
                agent_turns[agent_name] += 1
                debate_history.append({
                    "round": 1,
                    "agent": agent_name,
                    "turn": agent_turns[agent_name],
                    "message": response.get("analysis", ""),
                    "timestamp": response.get("timestamp", datetime.now(timezone.utc).isoformat())
                })
                logger.info(f"{agent_name} completed turn {agent_turns[agent_name]}")
            else:
                logger.warning(f"Failed to get initial response from {agent_name}: {response.get('error', 'Unknown error')}")
        
        # Structured debate rounds with strict turn-taking
        round_num = 2
        while round_num <= max_rounds:
            logger.info(f"=== DEBATE ROUND {round_num} ===")
            
            # Check if all agents have spoken at least min_turns_per_agent times
            min_turns_met = all(turns >= min_turns_per_agent for turns in agent_turns.values())
            
            # Check for consensus (simplified consensus check)
            if round_num > 2 and min_turns_met:
                consensus_reached = self._check_consensus(current_positions)
                if consensus_reached:
                    logger.info(f"Consensus reached after {round_num-1} rounds")
                    break
            
            # Conduct round with strict turn-taking
            round_completed = await self._conduct_debate_round(
                round_num, required_agents, current_positions, debate_history,
                agent_turns, context_id, task_id, metadata
            )
            
            if not round_completed:
                logger.warning(f"Round {round_num} could not be completed, ending debate")
                break
                
            round_num += 1
        
        # Ensure minimum participation requirement
        agents_with_insufficient_turns = [
            agent for agent, turns in agent_turns.items() 
            if turns < min_turns_per_agent
        ]
        
        if agents_with_insufficient_turns:
            logger.info(f"=== ADDITIONAL TURNS FOR MINIMUM PARTICIPATION ===")
            for agent_name in agents_with_insufficient_turns:
                while agent_turns[agent_name] < min_turns_per_agent:
                    await self._conduct_additional_turn(
                        agent_name, current_positions, debate_history,
                        agent_turns, context_id, task_id, metadata
                    )
        
        # Convert debate results to agent response format
        final_responses = []
        for agent_name in required_agents:
            agent_url = self._agent_urls.get(agent_name)
            if agent_url and agent_name in current_positions:
                final_responses.append({
                    "agent": agent_name,
                    "success": True,
                    "analysis": current_positions[agent_name],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "debate_turns": agent_turns[agent_name],
                    "debate_metadata": {
                        "total_rounds": round_num - 1,
                        "total_turns": sum(agent_turns.values()),
                        "agent_participation": agent_turns
                    }
                })
        
        # Log debate summary
        logger.info(f"Structured debate completed:")
        logger.info(f"- Total rounds: {round_num - 1}")
        logger.info(f"- Agent participation: {agent_turns}")
        logger.info(f"- Total turns: {sum(agent_turns.values())}")
        
        return final_responses

    async def _conduct_debate_round(
        self,
        round_num: int,
        required_agents: List[str],
        current_positions: Dict[str, str],
        debate_history: List[Dict[str, Any]],
        agent_turns: Dict[str, int],
        context_id: str,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Conduct a single debate round with strict turn-taking."""
        
        # Round-robin turn order
        for agent_name in required_agents:
            agent_url = self._agent_urls.get(agent_name)
            if not agent_url:
                continue
            
            # Prepare debate context for this agent
            other_positions = {k: v for k, v in current_positions.items() if k != agent_name}
            debate_prompt = self._create_debate_prompt(
                round_num, agent_name, other_positions, debate_history
            )
            
            message = {
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": debate_prompt
                    }
                ],
                "contextId": context_id,
                "taskId": task_id
            }
            
            logger.info(f"Round {round_num}: {agent_name}'s turn (turn #{agent_turns[agent_name] + 1})")
            response = await self._send_message_to_agent(agent_name, agent_url, message, metadata)
            
            if response.get("success", False):
                # Update agent's position
                current_positions[agent_name] = response.get("analysis", "")
                agent_turns[agent_name] += 1
                
                # Record in debate history
                debate_history.append({
                    "round": round_num,
                    "agent": agent_name,
                    "turn": agent_turns[agent_name],
                    "message": response.get("analysis", ""),
                    "timestamp": response.get("timestamp", datetime.now(timezone.utc).isoformat())
                })
                
                logger.info(f"{agent_name} completed turn {agent_turns[agent_name]}")
            else:
                logger.warning(f"Failed to get response from {agent_name} in round {round_num}: {response.get('error', 'Unknown error')}")
                return False
        
        return True

    async def _conduct_additional_turn(
        self,
        agent_name: str,
        current_positions: Dict[str, str],
        debate_history: List[Dict[str, Any]],
        agent_turns: Dict[str, int],
        context_id: str,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Conduct additional turn to meet minimum participation requirement."""
        
        agent_url = self._agent_urls.get(agent_name)
        if not agent_url:
            return False
        
        other_positions = {k: v for k, v in current_positions.items() if k != agent_name}
        prompt = f"""
ADDITIONAL PARTICIPATION ROUND - Turn #{agent_turns[agent_name] + 1}

You need to provide additional input to meet the minimum participation requirement of the structured debate.

CURRENT POSITIONS FROM OTHER AGENTS:
{chr(10).join([f"{agent.upper()}: {pos}" for agent, pos in other_positions.items()])}

YOUR CURRENT POSITION:
{current_positions.get(agent_name, "No position recorded")}

Please provide any additional analysis, refinements to your position, or final thoughts to contribute to the collaborative decision-making process.
"""
        
        message = {
            "kind": "message",
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": prompt
                }
            ],
            "contextId": context_id,
            "taskId": task_id
        }
        
        logger.info(f"Additional turn for {agent_name} (turn #{agent_turns[agent_name] + 1})")
        response = await self._send_message_to_agent(agent_name, agent_url, message, metadata)
        
        if response.get("success", False):
            current_positions[agent_name] = response.get("analysis", "")
            agent_turns[agent_name] += 1
            
            debate_history.append({
                "round": "additional",
                "agent": agent_name,
                "turn": agent_turns[agent_name],
                "message": response.get("analysis", ""),
                "timestamp": response.get("timestamp", datetime.now(timezone.utc).isoformat())
            })
            
            logger.info(f"{agent_name} completed additional turn {agent_turns[agent_name]}")
            return True
        else:
            logger.warning(f"Failed to get additional response from {agent_name}: {response.get('error', 'Unknown error')}")
            return False

    def _create_debate_prompt(
        self,
        round_num: int,
        current_agent: str,
        other_positions: Dict[str, str],
        debate_history: List[Dict[str, Any]]  # Currently unused but kept for future enhancements
    ) -> str:
        """Create debate prompt for current agent's turn."""
        
        prompt = f"""
STRUCTURED DEBATE - ROUND {round_num} - {current_agent.upper()} AGENT TURN

You are participating in a structured multi-agent debate to reach consensus on an investment decision.

OTHER AGENTS' CURRENT POSITIONS:
{chr(10).join([f"{agent.upper()}: {pos}" for agent, pos in other_positions.items()])}

INSTRUCTIONS FOR YOUR TURN:
1. Review the positions from other agents above
2. Identify any points of agreement or disagreement
3. Present evidence and reasoning that supports your analysis
4. Address specific concerns or counterarguments raised by other agents
5. Refine your position if convinced by other agents' arguments
6. Stay focused on reaching the best collective investment decision

This is a collaborative process. Your goal is not to "win" but to contribute to finding the most accurate analysis through reasoned discussion.

Provide your response for this turn of the debate:
"""
        
        return prompt

    def _check_consensus(self, current_positions: Dict[str, str]) -> bool:
        """
        Check if consensus has been reached among agents.
        Simple implementation - can be enhanced with more sophisticated analysis.
        """
        if not current_positions or len(current_positions) < 2:
            return False
        
        # Extract sentiment indicators from each position
        sentiments = []
        for position in current_positions.values():
            position_lower = position.lower()
            if any(word in position_lower for word in ["buy", "bullish", "positive", "recommend buy"]):
                sentiments.append("buy")
            elif any(word in position_lower for word in ["sell", "bearish", "negative", "recommend sell"]):
                sentiments.append("sell")
            else:
                sentiments.append("hold")
        
        # Check if majority agrees
        if not sentiments:
            return False
        
        most_common = max(set(sentiments), key=sentiments.count)
        consensus_ratio = sentiments.count(most_common) / len(sentiments)
        
        # Require at least 75% agreement for consensus
        return consensus_ratio >= 0.75

    def _consolidate_analyses(
        self, 
        agent_responses: List[Dict[str, Any]], 
        user_message: str
    ) -> str:
        """Use LLM to consolidate multiple agent analyses into coherent response."""
        
        # Check if this was a structured debate
        debate_occurred = any(
            r.get("debate_metadata") is not None 
            for r in agent_responses 
            if r.get("success", False)
        )
        
        if debate_occurred:
            system_prompt = """You are a financial analysis consolidator specializing in multi-agent debate synthesis. Your role is to consolidate the results of a structured debate between specialist agents into a coherent, actionable response.

CONSOLIDATION GUIDELINES FOR DEBATE RESULTS:
1. Acknowledge that the analysis came from a structured multi-agent debate process
2. Synthesize the final positions reached through debate and discussion
3. Highlight areas where consensus was achieved and explain the reasoning
4. Address any remaining disagreements and explain the different perspectives
5. Provide clear, actionable recommendations based on the collective reasoning
6. Emphasize the collaborative nature of the analysis and increased confidence from multiple expert perspectives
7. Keep response focused and relevant to the user's original question

Format your response as a comprehensive financial analysis that emphasizes the rigorous debate process used to reach these conclusions."""
        else:
            system_prompt = """You are a financial analysis consolidator. Your role is to synthesize insights from multiple specialist agents into a coherent, actionable response.

CONSOLIDATION GUIDELINES:
1. Integrate insights from all successful agent responses
2. Identify agreements and disagreements between agents
3. Provide clear, actionable recommendations when possible
4. Highlight key risks and opportunities
5. If agents disagree, explain the different perspectives
6. Keep response focused and relevant to the user's original question

Format your response as a comprehensive financial analysis that addresses the user's request."""

        try:
            # Prepare agent analyses for LLM
            analyses_text = []
            successful_responses = [r for r in agent_responses if r.get("success", False)]
            
            for response in successful_responses:
                agent_name = response.get("agent", "unknown")
                analysis = response.get("analysis", "No analysis provided")
                debate_info = ""
                
                # Add debate metadata if available
                if debate_occurred and response.get("debate_metadata"):
                    metadata = response["debate_metadata"]
                    debate_info = f" (Participated in {metadata.get('total_rounds', 'N/A')} debate rounds, {response.get('debate_turns', 0)} total turns)"
                
                analyses_text.append(f"**{agent_name.upper()} AGENT{debate_info}:**\n{analysis}")
            
            if not analyses_text:
                return "I apologize, but I was unable to get responses from the specialist agents. Please try again later."
            
            # Add debate process information if applicable
            debate_context = ""
            if debate_occurred:
                total_rounds = 0
                total_turns = 0
                for response in successful_responses:
                    if response.get("debate_metadata"):
                        total_rounds = max(total_rounds, response["debate_metadata"].get("total_rounds", 0))
                        total_turns = max(total_turns, response["debate_metadata"].get("total_turns", 0))
                
                debate_context = f"""
DEBATE PROCESS INFORMATION:
- Structured multi-agent debate was conducted to ensure thorough analysis
- Total debate rounds: {total_rounds}
- Total agent turns: {total_turns}
- Each agent participated at least twice to meet AlphaAgents paper requirements
- Strict turn-taking prevented any single agent from dominating the discussion

"""
            
            consolidation_prompt = f"""
USER REQUEST: {user_message}
{debate_context}
AGENT ANALYSES:
{chr(10).join(analyses_text)}

Please provide a consolidated analysis that synthesizes these perspectives into a coherent response to the user's request.
"""
            
            # Log the complete consolidation prompt for detailed analysis
            full_consolidation_prompt = f"SYSTEM: {system_prompt}\n\nHUMAN: {consolidation_prompt}"
            logger.info(f"🤖 LLM CONSOLIDATION PROMPT:\n{full_consolidation_prompt}")
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=consolidation_prompt)
            ])
            
            # Log the complete consolidation response
            logger.info(f"🤖 LLM CONSOLIDATION RESPONSE: {response.content}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error consolidating analyses: {e}")
            # Fallback: return raw analyses
            successful_responses = [r for r in agent_responses if r.get("success", False)]
            if successful_responses:
                consolidated = "Based on specialist agent analyses:\n\n"
                for response in successful_responses:
                    agent_name = response.get("agent", "unknown")
                    analysis = response.get("analysis", "No analysis")
                    consolidated += f"**{agent_name.title()} Analysis:**\n{analysis}\n\n"
                return consolidated
            else:
                return "I was unable to get responses from the specialist agents. Please try again later."

    async def process_message(
        self,
        message: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an incoming A2A message and coordinate with specialist agents.
        
        Args:
            message: A2A Message object
            metadata: Optional metadata from the request
            
        Returns:
            A2A-compliant response (Task or Message)
        """
        
        try:
            # Extract message content
            parts = message.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
            user_message = " ".join(text_parts).strip()
            
            if not user_message:
                return {
                    "kind": "message",
                    "messageId": str(uuid.uuid4()),
                    "role": "agent",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "I received your message, but it appears to be empty. Please provide a financial analysis request."
                        }
                    ],
                    "contextId": message.get("contextId", str(uuid.uuid4()))
                }
            
            context_id = message.get("contextId", str(uuid.uuid4()))
            task_id = message.get("taskId", str(uuid.uuid4()))
            
            logger.info(f"Processing user message: {user_message[:100]}...")
            
            # Extract and enhance metadata with risk tolerance
            enhanced_metadata = self._enhance_metadata_with_risk_tolerance(user_message, metadata)
            
            # Coordinate with specialist agents
            agent_responses = await self._coordinate_agents(
                user_message, context_id, task_id, enhanced_metadata
            )
            
            # Consolidate responses using LLM
            consolidated_analysis = self._consolidate_analyses(agent_responses, user_message)
            
            # Return A2A-compliant message response
            return {
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": consolidated_analysis
                    }
                ],
                "contextId": context_id,
                "metadata": {
                    "agent_responses_summary": {
                        "total_agents_consulted": len(agent_responses),
                        "successful_responses": len([r for r in agent_responses if r.get("success", False)]),
                        "agents_used": [r.get("agent") for r in agent_responses if r.get("success", False)]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "role": "agent", 
                "parts": [
                    {
                        "kind": "text",
                        "text": f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again."
                    }
                ],
                "contextId": message.get("contextId", str(uuid.uuid4()))
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of GroupChat agent and specialist agents."""
        try:
            # Check registry connectivity
            registry_health = await self.registry_service.health_check()
            
            # Try to fetch agent URLs
            if not await self._ensure_agent_urls():
                return {
                    "status": "unhealthy",
                    "error": "Failed to fetch specialist agent URLs from registry",
                    "registry_health": registry_health
                }
            
            # Check specialist agent health
            agent_health = {}
            for agent_name, agent_url in self._agent_urls.items():
                try:
                    health_url = f"{agent_url.rstrip('/')}/health"
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
                        async with session.get(health_url) as response:
                            agent_health[agent_name] = {
                                "status": "healthy" if response.status == 200 else "unhealthy",
                                "status_code": response.status,
                                "url": health_url
                            }
                except Exception as e:
                    agent_health[agent_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "url": agent_url
                    }
            
            healthy_agents = sum(1 for h in agent_health.values() if h.get("status") == "healthy")
            total_agents = len(agent_health)
            
            return {
                "status": "healthy" if healthy_agents == total_agents else "degraded",
                "registry_health": registry_health,
                "specialist_agents": agent_health,
                "summary": {
                    "total_agents": total_agents,
                    "healthy_agents": healthy_agents,
                    "failed_agents": total_agents - healthy_agents
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }