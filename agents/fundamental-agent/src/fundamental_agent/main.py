"""Main entry point for the Fundamental Analysis Agent A2A server."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from fundamental_agent.server import create_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fundamental-agent.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main function to start the Fundamental Analysis Agent server."""
    try:
        # Load environment variables
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f"No .env file found at {env_path}")
        
        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "3003"))
        log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        logger.info("Starting Fundamental Agent...")
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Log Level: {log_level}")
        
        # Create and start the server
        server = await create_server(host=host, port=port)
        
        logger.info("Fundamental Agent server starting...")
        await server.start_server()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())