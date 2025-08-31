"""Agent Registry service for dynamic agent discovery."""

import logging
from typing import Dict, List, Optional
import httpx
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AgentRegistryService:
    """Service for interacting with the Agent Registry."""
    
    def __init__(self, registry_url: str, timeout: float = 10.0):
        """Initialize the Agent Registry service."""
        self.registry_url = registry_url.rstrip('/')
        self.timeout = timeout
        self._agent_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    async def get_agent_by_name(self, name: str) -> Optional[Dict]:
        """Get agent information by name from the registry."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.registry_url}/agents",
                    params={"name": name}
                )
                response.raise_for_status()
                
                agents = response.json()
                
                if not agents:
                    logger.warning(f"No agents found with name '{name}'")
                    return None
                
                if len(agents) > 1:
                    logger.warning(
                        f"Multiple agents found with name '{name}', using the first one"
                    )
                
                agent = agents[0]
                logger.info(f"Found agent '{name}' at {agent.get('url', 'unknown URL')}")
                return agent
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting agent '{name}': {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error getting agent '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting agent '{name}': {e}")
            return None
    
    async def get_specialist_agents(
        self,
        valuation_name: str = "valuation",
        sentiment_name: str = "sentiment", 
        fundamental_name: str = "fundamental"
    ) -> Dict[str, Optional[str]]:
        """Get URLs for all specialist agents."""
        logger.info("Fetching specialist agent URLs from registry...")
        
        # Check cache first
        if self._is_cache_valid():
            logger.info("Using cached agent URLs")
            return self._agent_cache.copy()
        
        # Fetch agents concurrently
        tasks = [
            self.get_agent_by_name(valuation_name),
            self.get_agent_by_name(sentiment_name),
            self.get_agent_by_name(fundamental_name)
        ]
        
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract URLs from results
        agent_urls = {}
        agent_names = ["valuation", "sentiment", "fundamental"]
        
        for i, (agent_name, result) in enumerate(zip(agent_names, agent_results)):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {agent_name} agent: {result}")
                agent_urls[agent_name] = None
            elif result is None:
                logger.error(f"Agent '{agent_name}' not found in registry")
                agent_urls[agent_name] = None
            else:
                url = result.get("url")
                if url:
                    agent_urls[agent_name] = url
                    logger.info(f"✓ {agent_name} agent: {url}")
                else:
                    logger.error(f"No URL found for {agent_name} agent")
                    agent_urls[agent_name] = None
        
        # Update cache
        self._agent_cache = agent_urls.copy()
        self._cache_timestamp = datetime.now(timezone.utc)
        
        return agent_urls
    
    def _is_cache_valid(self) -> bool:
        """Check if the agent cache is still valid."""
        if not self._cache_timestamp or not self._agent_cache:
            return False
        
        age = (datetime.now(timezone.utc) - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl
    
    def clear_cache(self):
        """Clear the agent URL cache."""
        self._agent_cache = {}
        self._cache_timestamp = None
        logger.info("Agent cache cleared")
    
    async def health_check(self) -> Dict:
        """Check if the Agent Registry is accessible."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.registry_url}/health")
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "registry_url": self.registry_url,
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "registry_url": self.registry_url,
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            return {
                "status": "unreachable",
                "registry_url": self.registry_url,
                "error": str(e)
            }
    
    async def list_all_agents(self) -> List[Dict]:
        """List all agents from the registry."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.registry_url}/agents")
                response.raise_for_status()
                
                agents = response.json()
                logger.info(f"Found {len(agents)} agents in registry")
                return agents
                
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return []