"""A2A JSON-RPC Server for the Fundamental Analysis Agent."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import FundamentalAgent

logger = logging.getLogger(__name__)


# A2A Protocol Models
class A2AMessage(BaseModel):
    """A2A protocol message structure."""
    kind: str = "message"
    messageId: str
    role: str
    parts: List[Dict[str, Any]]
    contextId: Optional[str] = None
    taskId: Optional[str] = None


class A2ATaskStatus(BaseModel):
    """A2A task status structure."""
    state: str  # "pending", "running", "completed", "failed"
    message: Optional[A2AMessage] = None
    error: Optional[str] = None
    progress: Optional[float] = None


class A2ATask(BaseModel):
    """A2A task structure."""
    id: str
    status: A2ATaskStatus
    metadata: Optional[Dict[str, Any]] = None


class MessageSendRequest(BaseModel):
    """Request model for message/send endpoint."""
    message: A2AMessage
    metadata: Optional[Dict[str, Any]] = None


class MessageSendResponse(BaseModel):
    """Response model for message/send endpoint."""
    id: str
    result: A2ATask


class TasksGetRequest(BaseModel):
    """Request model for tasks/get endpoint."""
    id: str


class TasksGetResponse(BaseModel):
    """Response model for tasks/get endpoint."""
    result: A2ATask


class A2AFundamentalServer:
    """A2A JSON-RPC server for the Fundamental Analysis Agent."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 3003,
        openai_api_key: Optional[str] = None
    ):
        """Initialize the A2A server.
        
        Args:
            host: Server host address
            port: Server port number  
            openai_api_key: OpenAI API key (if not provided, will use env var)
        """
        self.host = host
        self.port = port
        
        # Initialize the fundamental agent
        try:
            self.agent = FundamentalAgent(openai_api_key=openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize fundamental agent: {str(e)}")
            raise
        
        # Task storage (in production, use proper database)
        self.tasks: Dict[str, A2ATask] = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Fundamental Analysis Agent A2A Server",
            description="A2A-compliant server for fundamental analysis using LLMs and financial data",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"A2A Fundamental Server initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup FastAPI routes for A2A protocol."""
        
        @self.app.post("/")
        async def json_rpc_handler(request: Request):
            """Handle JSON-RPC requests according to A2A protocol."""
            try:
                body = await request.json()
                
                # Validate JSON-RPC structure
                if not isinstance(body, dict) or "method" not in body:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32600,
                                "message": "Invalid Request"
                            },
                            "id": body.get("id")
                        }
                    )
                
                method = body.get("method")
                params = body.get("params", {})
                request_id = body.get("id")
                
                # Route to appropriate handler
                if method == "message/send":
                    result = await self._handle_message_send(params)
                elif method == "tasks/get":
                    result = await self._handle_tasks_get(params)
                else:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}"
                            },
                            "id": request_id
                        }
                    )
                
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": request_id
                    }
                )
                
            except Exception as e:
                logger.error(f"JSON-RPC handler error: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        },
                        "id": body.get("id") if isinstance(body, dict) else None
                    }
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "agent": "fundamental-agent",
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/.well-known/agent-card")
        async def get_agent_card(request: Request):
            """Return the agent card."""
            # Load the agent card from file
            try:
                card_path = os.path.join(os.path.dirname(__file__), "../../fundamental-agent-card.json")
                with open(card_path, "r") as f:
                    response_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading agent card: {str(e)}")
                response_data = {"error": "Could not load agent card"}
            
            # Set CORS headers only for this route
            response = JSONResponse(content=response_data)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response

    async def _handle_message_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message/send requests."""
        try:
            # Validate request structure
            if "message" not in params:
                raise ValueError("Missing 'message' in request parameters")
            
            message_data = params["message"]
            metadata = params.get("metadata", {})
            
            # Create A2A message object
            message = A2AMessage(**message_data)
            
            # Extract user input from message parts
            user_input = ""
            for part in message.parts:
                if part.get("kind") == "text":
                    user_input += part.get("text", "")
            
            if not user_input.strip():
                raise ValueError("No text content found in message")
            
            # Create task
            task_id = str(uuid.uuid4())
            
            # Initialize task as running
            task = A2ATask(
                id=task_id,
                status=A2ATaskStatus(state="running"),
                metadata=metadata
            )
            self.tasks[task_id] = task
            
            # Process the request asynchronously
            asyncio.create_task(self._process_fundamental_analysis(task_id, user_input, metadata))
            
            return {
                "id": task_id,
                "status": task.status.dict()
            }
            
        except Exception as e:
            logger.error(f"Error handling message/send: {str(e)}")
            # Create failed task
            task_id = str(uuid.uuid4())
            failed_message = A2AMessage(
                messageId=str(uuid.uuid4()),
                role="assistant",
                parts=[{
                    "kind": "text",
                    "text": f"Analysis failed: {str(e)}"
                }]
            )
            
            task = A2ATask(
                id=task_id,
                status=A2ATaskStatus(
                    state="failed",
                    message=failed_message,
                    error=str(e)
                )
            )
            self.tasks[task_id] = task
            
            return {
                "id": task_id,
                "status": task.status.dict()
            }

    async def _handle_tasks_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/get requests."""
        try:
            if "id" not in params:
                raise ValueError("Missing 'id' in request parameters")
            
            task_id = params["id"]
            
            if task_id not in self.tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self.tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.dict(),
                "metadata": task.metadata
            }
            
        except Exception as e:
            logger.error(f"Error handling tasks/get: {str(e)}")
            raise

    async def _process_fundamental_analysis(
        self,
        task_id: str,
        user_input: str,
        metadata: Dict[str, Any]
    ):
        """Process fundamental analysis request asynchronously."""
        try:
            # Extract parameters from metadata
            risk_tolerance = metadata.get("risk_tolerance", "neutral")
            analysis_depth = metadata.get("analysis_depth", "comprehensive")
            focus_areas = metadata.get("focus_areas", [])
            
            # Determine analysis type based on user input
            analysis_result = None
            
            # Check if it's a sector comparison request
            if any(word in user_input.lower() for word in ["compare", "comparison", "vs", "versus"]):
                # Use LLM-based ticker extraction for robust parsing
                tickers = await self._extract_tickers_from_text_llm(user_input)
                logger.info(f"Comparison request - extracted tickers: {tickers}")
                
                if len(tickers) > 1:
                    # Ora siamo sicuri che i ticker sono validi e ≤5
                    analysis_result = await asyncio.to_thread(
                        self.agent.sector_comparison,
                        tickers,
                        "fundamental_metrics",
                        risk_tolerance
                    )
                else:
                    logger.info("Insufficient tickers for comparison, using comprehensive analysis")
                    analysis_result = await asyncio.to_thread(
                        self.agent.comprehensive_analysis,
                        user_input,
                        risk_tolerance
                    )
            
            # Check if it's a quick valuation request
            elif any(word in user_input.lower() for word in ["valuation", "value", "price target", "dcf"]):
                tickers = await self._extract_tickers_from_text_llm(user_input)
                if tickers:
                    valuation_method = "dcf" if "dcf" in user_input.lower() else "hybrid"
                    analysis_result = await asyncio.to_thread(
                        self.agent.quick_valuation,
                        tickers[0],
                        valuation_method,
                        risk_tolerance
                    )
            
            # Default to comprehensive fundamental analysis
            if not analysis_result:
                analysis_result = await asyncio.to_thread(
                    self.agent.analyze_fundamental,
                    user_input,
                    risk_tolerance,
                    analysis_depth,
                    focus_areas if focus_areas else None
                )
            
            # Create response message
            if analysis_result and analysis_result.get("success"):
                analysis_text = analysis_result.get("analysis", "Analysis completed successfully")
                response_message = A2AMessage(
                    messageId=str(uuid.uuid4()),
                    role="assistant", 
                    parts=[{
                        "kind": "text",
                        "text": analysis_text
                    }]
                )
                
                # Update task status
                self.tasks[task_id].status = A2ATaskStatus(
                    state="completed",
                    message=response_message
                )
            else:
                # Analysis failed
                error_message = analysis_result.get("error", "Unknown analysis error") if analysis_result else "Analysis failed"
                response_message = A2AMessage(
                    messageId=str(uuid.uuid4()),
                    role="assistant",
                    parts=[{
                        "kind": "text", 
                        "text": f"Fundamental analysis failed: {error_message}"
                    }]
                )
                
                self.tasks[task_id].status = A2ATaskStatus(
                    state="failed",
                    message=response_message,
                    error=error_message
                )
            
        except Exception as e:
            logger.error(f"Error processing fundamental analysis for task {task_id}: {str(e)}")
            
            # Update task with error
            error_message = A2AMessage(
                messageId=str(uuid.uuid4()),
                role="assistant",
                parts=[{
                    "kind": "text",
                    "text": f"Analysis encountered an error: {str(e)}"
                }]
            )
            
            self.tasks[task_id].status = A2ATaskStatus(
                state="failed",
                message=error_message,
                error=str(e)
            )

    async def _extract_tickers_from_text_llm(self, text: str) -> List[str]:
        """Extract stock tickers using LLM for robust, multilingual parsing."""
        
        system_prompt = """You are a financial ticker extraction specialist.
        
        Your task is to identify ONLY company names and stock tickers from user text.
        
        EXTRACTION RULES:
        1. Look for publicly traded company names (e.g., "Tesla", "Apple", "Microsoft", "Palantir")
        2. Look for stock tickers (e.g., "AAPL", "TSLA", "MSFT", "PLTR") 
        3. Convert company names to their standard stock tickers when possible
        4. Ignore common words, articles, prepositions, and non-financial terms
        5. Work with any language (English, Italian, Spanish, French, etc.)
        6. Maximum 5 companies/tickers per response
        
        COMMON MAPPINGS:
        - Tesla → TSLA
        - Apple → AAPL  
        - Microsoft → MSFT
        - Palantir → PLTR
        - Google/Alphabet → GOOGL
        - Amazon → AMZN
        - Meta/Facebook → META
        - Netflix → NFLX
        - Nvidia → NVDA
        
        RESPONSE FORMAT:
        Return ONLY a JSON array of stock tickers, like: ["AAPL", "MSFT", "TSLA"]
        If no valid companies/tickers found, return: []
        
        EXAMPLES:
        "Should I invest in Apple or Microsoft?" → ["AAPL", "MSFT"]
        "Compare TSLA vs NVDA performance" → ["TSLA", "NVDA"] 
        "Sono indeciso se investire in Tesla oppure in Palantir" → ["TSLA", "PLTR"]
        "What is the market doing today?" → []
        "Analyze Amazon and Google fundamentals" → ["AMZN", "GOOGL"]"""
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage
            import os
            import json
            
            # Usa un modello veloce ed economico per l'extraction
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.0  # Deterministic per parsing
            )
            
            response = await asyncio.to_thread(
                llm.invoke,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Extract tickers from: {text}")
                ]
            )
            
            # Parse JSON response
            try:
                tickers = json.loads(response.content.strip())
                if isinstance(tickers, list):
                    # Valida che siano ticker ragionevoli (2-5 caratteri)
                    valid_tickers = [
                        t.upper() for t in tickers 
                        if isinstance(t, str) and 2 <= len(t) <= 5
                    ]
                    logger.info(f"LLM extracted tickers: {valid_tickers} from text: '{text[:100]}...'")
                    return valid_tickers[:5]  # Safety cap
                else:
                    logger.warning(f"LLM returned non-list: {response.content}")
                    return []
                    
            except json.JSONDecodeError:
                logger.error(f"LLM response not valid JSON: {response.content}")
                return []
                
        except Exception as e:
            logger.error(f"Error in LLM ticker extraction: {str(e)}")
            # Fallback al regex method esistente come backup
            return self._extract_tickers_from_text_regex_fallback(text)

    def _extract_tickers_from_text_regex_fallback(self, text: str) -> List[str]:
        """Fallback regex method - more conservative pattern."""
        import re
        
        # Pattern più conservativo: solo ticker di 2-5 caratteri maiuscoli
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Exclude common words più aggressivo
        exclude_words = {
            "AND", "OR", "THE", "FOR", "WITH", "FROM", "TO", "OF", "IN", "ON", 
            "AT", "BY", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "HAVE", "HAS", 
            "HAD", "DO", "DOES", "DID", "WILL", "WOULD", "COULD", "SHOULD", 
            "MAY", "MIGHT", "CAN", "GET", "GOT", "PUT", "SET", "NEW", "OLD", 
            "BIG", "LOW", "HIGH", "ALL", "ANY", "MORE", "MOST", "BEST", "GOOD", "BAD",
            # Parole italiane comuni
            "SONO", "COME", "ANNI", "SOLO", "MOLTO", "ANCHE", "DOVE", "OGNI", "DEVE"
        }
        
        tickers = [t for t in potential_tickers if t not in exclude_words]
        return list(dict.fromkeys(tickers))[:5]  # Rimuovi duplicati e limita a 5
        
    def _extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract potential stock tickers from text - Legacy method for compatibility."""
        # Chiama il nuovo metodo LLM-based in modo sincrono
        import asyncio
        try:
            # Se siamo già in un loop async, usa create_task
            if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
                task = asyncio.create_task(self._extract_tickers_from_text_llm(text))
                return asyncio.run_coroutine_threadsafe(task, asyncio.get_event_loop()).result()
            else:
                # Se non siamo in un loop, creane uno nuovo
                return asyncio.run(self._extract_tickers_from_text_llm(text))
        except Exception as e:
            logger.error(f"Error in async ticker extraction, falling back to regex: {str(e)}")
            return self._extract_tickers_from_text_regex_fallback(text)

    async def start_server(self):
        """Start the A2A server."""
        import uvicorn
        
        logger.info(f"Starting A2A Fundamental Server on http://{self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


async def create_server(
    host: str = None,
    port: int = None,
    openai_api_key: str = None
) -> A2AFundamentalServer:
    """Create and configure the A2A server."""
    
    # Use environment variables as defaults
    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", "3003"))
    
    server = A2AFundamentalServer(
        host=host,
        port=port,
        openai_api_key=openai_api_key
    )
    
    return server