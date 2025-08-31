"""GroupChat Agent for coordinating multi-agent debates and collaborative analysis."""

from .registry_service import AgentRegistryService
from .server import GroupChatServer, create_app, run_server
from .test_client import GroupChatTestClient, run_tests

__version__ = "1.0.0"
__author__ = "AlphaAgents Financial"

__all__ = [
    "AgentRegistryService",
    "GroupChatServer",
    "create_app",
    "run_server",
    "GroupChatTestClient",
    "run_tests"
]

