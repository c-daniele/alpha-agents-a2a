# Fundamental Agent

A specialized A2A-compliant agent for comprehensive stock fundamental analysis through financial reports and sector analysis. This agent is part of the AlphaAgents portfolio analysis system and provides in-depth stock valuation using 10-K and 10-Q reports, sector trends, and financial statement data.

## Features

- **Financial Report Analysis**: Deep analysis of 10-K and 10-Q SEC filings
- **RAG-based Analysis**: Retrieval-Augmented Generation for comprehensive fundamental analysis  
- **Sector Comparison**: Industry and sector trend analysis
- **Domain Expertise**: Built-in financial analysis expertise and best practices
- **A2A Protocol Compliant**: Full JSON-RPC 2.0 implementation with streaming support

## Installation

This agent uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the agent directory
cd agents/fundamental-agent

# Install dependencies
uv sync --native-tls
```

## Configuration

Create a `.env` file in the agent directory:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# SEC API Configuration (optional)
SEC_API_KEY=your_sec_api_key_here

# Server Configuration  
HOST=0.0.0.0
PORT=3003

# Agent Configuration
AGENT_NAME=fundamental-agent
AGENT_VERSION=1.0.0
LOG_LEVEL=INFO
```

## Usage

### Starting the Agent Server

```bash
# Run with uv
uv run python src/fundamental_agent/main.py

# Or activate the virtual environment first
uv shell
python src/fundamental_agent/main.py
```

The agent will start an A2A-compliant JSON-RPC server on the configured host and port (default: `http://localhost:3003`).

## API Endpoints

- **POST /**: Main JSON-RPC endpoint for A2A protocol
- **GET /health**: Health check endpoint
- **GET /agent-card**: Returns the agent's capabilities card

## Capabilities

### Skills

1. **Financial Statement Analysis**
   - Balance sheet analysis and trend identification
   - Income statement analysis and profitability assessment
   - Cash flow statement analysis and liquidity evaluation

2. **SEC Filing Analysis**
   - 10-K annual report comprehensive analysis
   - 10-Q quarterly report analysis
   - Management discussion and analysis (MD&A) insights

3. **Sector and Industry Analysis**
   - Industry comparison and benchmarking
   - Sector trend identification
   - Competitive positioning analysis

4. **Fundamental Valuation**
   - DCF modeling and intrinsic value estimation
   - Multiple-based valuation (P/E, P/B, EV/EBITDA)
   - Growth analysis and sustainability assessment

### Tools

The agent has access to these specialized tools:

- **Finance Report Pull**: Enhanced yfinance API calls with validation and data retrieval
- **RAG Tool**: Domain expertise-guided fundamental analysis with document processing

## Integration

This agent is designed to work as part of the AlphaAgents multi-agent system:

1. **Standalone Operation**: Can be used independently for fundamental analysis
2. **GroupChat Integration**: Works with the GroupChat agent for multi-agent collaboration  
3. **Portfolio System**: Integrates with other agents (Sentiment, Valuation) for comprehensive analysis