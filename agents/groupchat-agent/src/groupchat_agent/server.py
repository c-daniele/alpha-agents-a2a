"""A2A JSON-RPC 2.0 Server for GroupChat Agent."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from .a2a_agent import A2AGroupChatAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request model."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    method: str = Field(description="Method name to call")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    id: Optional[str] = Field(default=None, description="Request ID")


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response model."""
    jsonrpc: str = Field(default="2.0")
    result: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[Dict[str, Any]] = Field(default=None)
    id: Optional[str] = Field(default=None)


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error model."""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


# JSON-RPC Error Codes
class ErrorCodes:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AGENT_ERROR = -32000


class GroupChatServer:
    """A2A JSON-RPC 2.0 Server for GroupChat Agent coordination."""

    def __init__(self):
        """Initialize the GroupChat server."""
        self.app = FastAPI(
            title="GroupChat Agent A2A Server",
            description="Multi-agent coordination and debate server for collaborative stock analysis",
            version="1.0.0"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize A2A GroupChat agent
        try:
            self.groupchat_agent = A2AGroupChatAgent(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                registry_url=os.getenv("AGENT_REGISTRY_URL"),
                valuation_agent_name=os.getenv("VALUATION_AGENT_NAME", "valuation"),
                sentiment_agent_name=os.getenv("SENTIMENT_AGENT_NAME", "sentiment"),
                fundamental_agent_name=os.getenv("FUNDAMENTAL_AGENT_NAME", "fundamental")
            )
            logger.info("A2A GroupChat agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GroupChat agent: {e}")
            raise

        # Set up routes
        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes for JSON-RPC endpoints."""

        @self.app.post("/")
        async def handle_jsonrpc(request: Request):
            """Handle JSON-RPC 2.0 requests."""
            try:
                # Parse JSON request
                try:
                    body = await request.json()
                except Exception as e:
                    return JSONRPCResponse(
                        error=JSONRPCError(
                            code=ErrorCodes.PARSE_ERROR,
                            message="Parse error",
                            data={"details": str(e)}
                        ).__dict__
                    ).__dict__

                # Validate JSON-RPC request
                try:
                    rpc_request = JSONRPCRequest(**body)
                except ValidationError as e:
                    return JSONRPCResponse(
                        error=JSONRPCError(
                            code=ErrorCodes.INVALID_REQUEST,
                            message="Invalid Request",
                            data={"validation_errors": e.errors()}
                        ).__dict__,
                        id=body.get("id")
                    ).__dict__

                # Route to appropriate method
                result = await self._route_method(rpc_request)

                # Check if result contains an error
                if isinstance(result, dict) and "error" in result:
                    return JSONRPCResponse(
                        error=result["error"],
                        id=rpc_request.id
                    ).__dict__

                # Return successful response
                return JSONRPCResponse(
                    result=result,
                    id=rpc_request.id
                ).__dict__

            except Exception as e:
                logger.error(f"Internal server error: {e}")
                return JSONRPCResponse(
                    error=JSONRPCError(
                        code=ErrorCodes.INTERNAL_ERROR,
                        message="Internal error",
                        data={"details": str(e)}
                    ).__dict__,
                    id=body.get("id") if 'body' in locals() else None
                ).__dict__

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                agent_health = await self.groupchat_agent.health_check()
                return {
                    "status": agent_health.get("status", "unknown"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_health": agent_health,
                    "server_info": {
                        "name": os.getenv("AGENT_NAME", "groupchat-agent"),
                        "version": os.getenv("AGENT_VERSION", "1.0.0")
                    }
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        
        @self.app.get("/.well-known/agent-card")
        async def get_agent_card(request: Request):
            """Return the agent card."""
            # Load the agent card from file
            try:
                card_path = os.path.join(os.path.dirname(__file__), "../../groupchat-agent-card.json")
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

        @self.app.get("/info")
        async def server_info():
            """Get server information and configuration."""
            return {
                "server_name": "GroupChat Agent A2A Server",
                "version": "1.0.0",
                "available_methods": [
                    "message/send",
                    "message/stream",
                    "tasks/get",
                    "tasks/cancel"
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _route_method(self, request: JSONRPCRequest) -> Dict[str, Any]:
        """Route JSON-RPC method to appropriate A2A agent function."""
        method_name = request.method
        params = request.params or {}

        try:
            if method_name == "message/send":
                return await self._handle_message_send(params)
            elif method_name == "message/stream":
                # Not implemented yet, return basic response
                return {"status": "streaming_not_implemented"}
            elif method_name == "tasks/get":
                # Not implemented yet, return empty tasks
                return {"tasks": []}
            elif method_name == "tasks/cancel":
                # Not implemented yet, return success
                return {"cancelled": True}
            else:
                raise Exception(f"Method '{method_name}' not found")

        except Exception as e:
            logger.error(f"Error executing method '{method_name}': {e}")
            # Return error response instead of raising an exception
            return {
                "error": {
                    "code": ErrorCodes.AGENT_ERROR,
                    "message": f"Agent method error: {str(e)}",
                    "data": {"method": method_name, "params": params}
                }
            }

    async def _handle_message_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle A2A message/send requests."""
        # Validate required parameters
        required_params = ["message"]
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            return {
                "error": {
                    "code": ErrorCodes.INVALID_PARAMS,
                    "message": f"Missing required parameters: {missing_params}",
                    "data": {"required": required_params, "provided": list(params.keys())}
                }
            }

        message = params["message"]
        metadata = params.get("metadata", {})

        logger.info(f"Processing A2A message/send request")

        # Convert string message to A2A Message format if needed
        if isinstance(message, str):
            a2a_message = {
                "kind": "message",
                "messageId": str(__import__('uuid').uuid4()),
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": message
                    }
                ],
                "contextId": metadata.get("session_id", str(__import__('uuid').uuid4()))
            }
        else:
            a2a_message = message

        # Process the message using the A2A agent
        result = await self.groupchat_agent.process_message(a2a_message, metadata)

        return result


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    server = GroupChatServer()
    return server.app


def run_server():
    """Run the GroupChat Agent server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3000"))

    logger.info(f"Starting GroupChat Agent server on {host}:{port}")

    uvicorn.run(
        "groupchat_agent.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    run_server()

