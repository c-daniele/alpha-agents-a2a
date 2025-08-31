"""Main entry point for the Valuation Agent."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv

from valuation_agent.server import A2AValuationServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('valuation-agent.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main function to start the Valuation Agent server."""
    
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning("No .env file found, using system environment variables")
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3001"))
    openai_api_key = os.getenv("OPENAI_API_KEY")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Validate required configuration
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        logger.error("Please set it in your .env file or system environment")
        sys.exit(1)
    
    logger.info(f"Starting Valuation Agent...")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Log Level: {log_level}")
    
    try:
        # Create and start the server
        server = A2AValuationServer(
            host=host,
            port=port,
            openai_api_key=openai_api_key
        )
        
        logger.info("Valuation Agent server starting...")
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())