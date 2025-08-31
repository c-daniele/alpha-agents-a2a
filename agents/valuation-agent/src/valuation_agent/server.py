"""A2A JSON-RPC Server implementation for the Valuation Agent."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uvicorn import Config, Server

import json
from fastapi.responses import JSONResponse

from .agent import ValuationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# A2A Protocol Models
class MessagePart(BaseModel):
    """Represents a part of a message (text, file, or data)."""
    kind: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    """Represents a message in the A2A protocol."""
    kind: str = "message"
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user" or "agent"
    parts: List[MessagePart]
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    referenceTaskIds: Optional[List[str]] = None
    extensions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    """Represents the status of a task."""
    state: str  # "submitted", "working", "completed", "failed", etc.
    timestamp: Optional[str] = None
    message: Optional[Message] = None


class Task(BaseModel):
    """Represents a task in the A2A protocol."""
    kind: str = "task"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    history: Optional[List[Message]] = None
    artifacts: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageSendParams(BaseModel):
    """Parameters for sending a message."""
    message: Message
    configuration: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request."""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class A2AValuationServer:
    """A2A JSON-RPC Server for the Valuation Agent."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 3001,
        openai_api_key: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize the valuation agent
        self.valuation_agent = ValuationAgent(openai_api_key=self.openai_api_key)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Valuation Agent A2A Server",
            description="A2A JSON-RPC server for stock valuation analysis",
            version="1.0.0"
        )
        
        # Store active tasks
        self.active_tasks: Dict[str, Task] = {}
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"A2A Valuation Server initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup FastAPI routes for JSON-RPC endpoints."""
        
        @self.app.post("/")
        async def json_rpc_endpoint(request: Request):
            """Main JSON-RPC endpoint."""
            try:
                body = await request.json()
                rpc_request = JSONRPCRequest(**body)
                
                # Route to appropriate method handler
                if rpc_request.method == "message/send":
                    return await self._handle_message_send(rpc_request)
                elif rpc_request.method == "message/stream":
                    return await self._handle_message_stream(rpc_request)
                elif rpc_request.method == "tasks/get":
                    return await self._handle_task_get(rpc_request)
                elif rpc_request.method == "tasks/cancel":
                    return await self._handle_task_cancel(rpc_request)
                else:
                    return JSONRPCResponse(
                        id=rpc_request.id,
                        error={
                            "code": -32601,
                            "message": "Method not found",
                            "data": f"Method '{rpc_request.method}' is not supported"
                        }
                    ).model_dump()
                    
            except Exception as e:
                logger.error(f"Error processing JSON-RPC request: {str(e)}")
                return JSONRPCResponse(
                    id=getattr(request, 'id', None),
                    error={
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    }
                ).model_dump()

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "agent": "valuation-agent", "version": "1.0.0"}

        @self.app.get("/.well-known/agent-card")
        async def get_agent_card(request: Request):
            """Return the agent card."""
            # Load the agent card from file
            try:
                card_path = os.path.join(os.path.dirname(__file__), "../../valuation-agent-card.json")
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

    async def _handle_message_send(self, rpc_request: JSONRPCRequest) -> Dict[str, Any]:
        """Handle message/send requests."""
        try:
            params = MessageSendParams(**rpc_request.params)
            message = params.message
            
            # Extract the user's query from message parts
            user_query = ""
            for part in message.parts:
                if part.kind == "text" and part.text:
                    user_query += part.text + " "
            user_query = user_query.strip()
            
            if not user_query:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error={
                        "code": -32602,
                        "message": "Invalid parameters",
                        "data": "No text content found in message"
                    }
                ).model_dump()
            
            # Create a new task
            task_id = message.taskId or str(uuid.uuid4())
            context_id = message.contextId or str(uuid.uuid4())
            
            # Create initial task with "working" status
            task = Task(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state="working",
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                history=[message],
                metadata={"agent_type": "valuation"}
            )
            
            self.active_tasks[task_id] = task
            
            # Process the analysis request
            try:
                # Extract analysis parameters from the query or metadata
                risk_tolerance = "neutral"  # default
                analysis_period_days = 365  # default
                
                # Check for risk tolerance in metadata or query
                if params.metadata:
                    risk_tolerance = params.metadata.get("risk_tolerance", risk_tolerance)
                    analysis_period_days = params.metadata.get("analysis_period_days", analysis_period_days)
                
                # Perform the valuation analysis
                analysis_result = await self.valuation_agent.analyze_stock(
                    stock_input=user_query,
                    risk_tolerance=risk_tolerance,
                    analysis_period_days=analysis_period_days
                )
                
                # Create agent response message
                agent_message = Message(
                    role="agent",
                    parts=[MessagePart(
                        kind="text",
                        text=analysis_result.get("analysis", "Analysis completed"),
                        metadata={
                            "analysis_success": analysis_result.get("success", False),
                            "agent_type": "valuation"
                        }
                    )],
                    contextId=context_id,
                    taskId=task_id
                )
                
                # Update task with completion
                task.status.state = "completed" if analysis_result.get("success") else "failed"
                task.status.timestamp = datetime.now(timezone.utc).isoformat()
                task.status.message = agent_message
                task.history.append(agent_message)
                
                # Return the completed task
                return JSONRPCResponse(
                    id=rpc_request.id,
                    result=task.model_dump()
                ).model_dump()
                
            except Exception as e:
                logger.error(f"Error in valuation analysis: {str(e)}")
                
                # Create error response message
                error_message = Message(
                    role="agent",
                    parts=[MessagePart(
                        kind="text",
                        text=f"I encountered an error while analyzing the stock: {str(e)}",
                        metadata={"error": True, "agent_type": "valuation"}
                    )],
                    contextId=context_id,
                    taskId=task_id
                )
                
                # Update task with failure
                task.status.state = "failed"
                task.status.timestamp = datetime.now(timezone.utc).isoformat()
                task.status.message = error_message
                task.history.append(error_message)
                
                return JSONRPCResponse(
                    id=rpc_request.id,
                    result=task.model_dump()
                ).model_dump()
                
        except Exception as e:
            logger.error(f"Error handling message/send: {str(e)}")
            return JSONRPCResponse(
                id=rpc_request.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            ).model_dump()

    async def _handle_message_stream(self, rpc_request: JSONRPCRequest) -> StreamingResponse:
        """Handle message/stream requests for streaming responses."""
        # For now, we'll implement basic streaming by chunking the regular response
        # In a full implementation, this would stream partial results as they're generated
        
        async def generate_stream():
            try:
                # Get the regular response
                response = await self._handle_message_send(rpc_request)
                
                # Convert to streaming format
                stream_data = f"data: {json.dumps(response)}\n\n"
                yield stream_data.encode('utf-8')
                
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": rpc_request.id,
                    "error": {
                        "code": -32603,
                        "message": "Streaming error",
                        "data": str(e)
                    }
                }
                stream_data = f"data: {json.dumps(error_response)}\n\n"
                yield stream_data.encode('utf-8')
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    async def _handle_task_get(self, rpc_request: JSONRPCRequest) -> Dict[str, Any]:
        """Handle tasks/get requests."""
        try:
            task_id = rpc_request.params.get("id") if rpc_request.params else None
            
            if not task_id:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error={
                        "code": -32602,
                        "message": "Invalid parameters",
                        "data": "Task ID is required"
                    }
                ).model_dump()
            
            if task_id in self.active_tasks:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    result=self.active_tasks[task_id].model_dump()
                ).model_dump()
            else:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error={
                        "code": -32001,
                        "message": "Task not found",
                        "data": f"Task with ID '{task_id}' not found"
                    }
                ).model_dump()
                
        except Exception as e:
            logger.error(f"Error handling tasks/get: {str(e)}")
            return JSONRPCResponse(
                id=rpc_request.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            ).model_dump()

    async def _handle_task_cancel(self, rpc_request: JSONRPCRequest) -> Dict[str, Any]:
        """Handle tasks/cancel requests."""
        try:
            task_id = rpc_request.params.get("id") if rpc_request.params else None
            
            if not task_id:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error={
                        "code": -32602,
                        "message": "Invalid parameters", 
                        "data": "Task ID is required"
                    }
                ).model_dump()
            
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                
                if task.status.state in ["completed", "failed", "canceled"]:
                    return JSONRPCResponse(
                        id=rpc_request.id,
                        error={
                            "code": -32002,
                            "message": "Task cannot be canceled",
                            "data": f"Task is already in '{task.status.state}' state"
                        }
                    ).model_dump()
                
                # Cancel the task
                task.status.state = "canceled"
                task.status.timestamp = datetime.now(timezone.utc).isoformat()
                
                return JSONRPCResponse(
                    id=rpc_request.id,
                    result=task.model_dump()
                ).model_dump()
            else:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error={
                        "code": -32001,
                        "message": "Task not found",
                        "data": f"Task with ID '{task_id}' not found"
                    }
                ).model_dump()
                
        except Exception as e:
            logger.error(f"Error handling tasks/cancel: {str(e)}")
            return JSONRPCResponse(
                id=rpc_request.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            ).model_dump()

    async def start(self):
        """Start the A2A server."""
        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )
        server = Server(config)
        
        logger.info(f"Starting A2A Valuation Server on http://{self.host}:{self.port}")
        await server.serve()


async def main():
    """Main function to start the server."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3001"))
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    # Create and start server
    server = A2AValuationServer(
        host=host,
        port=port,
        openai_api_key=openai_api_key
    )
    
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())