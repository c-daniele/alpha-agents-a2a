# Alpha Agents - Multi-Agent Portfolio Analysis System

This project starts from the paper [AlphaAgents](https://arxiv.org/abs/2508.11152v1) where they propose a multi-agent system for financial analysis and decision-making.

I've been inspired from the paper and tried to implement a comprehensive stock and portfolio analysis using Agent-to-Agent (A2A) protocol. The system consists of specialized agents that collaborate through structured debate mechanisms to provide in-depth financial analysis.

## Overview

Alpha Agents is designed to perform portfolio analysis and support stock selection through multiple specialized AI agents, each with domain-specific expertise:

- **Valuation Agent**: Technical analysis, volatility metrics, and risk assessment
- **Sentiment Agent**: News sentiment analysis and market perception evaluation
- **Fundamental Agent**: Financial statement analysis and sector comparison
- **GroupChat Agent**: Multi-agent coordination and collaborative analysis

## System Architecture

```
                    End Users
                        |
                GroupChat Agent
                   (Port 3000)
                        |
        +---------------+---------------+
        |               |               |
   Valuation        Sentiment      Fundamental
     Agent           Agent           Agent
  (Port 3001)     (Port 3002)     (Port 3003)
        |               |               |
   Yahoo Finance    News APIs       SEC Filings
   Historical      Multi-Source      10-K/10-Q
     Data          Collection        Analysis
```

## Features

### Core Capabilities

- **Multi-Agent Collaboration**: Structured debate mechanisms for consensus building
- **A2A Protocol Compliance**: Full JSON-RPC 2.0 implementation across all agents
- **Comprehensive Analysis**: Technical, fundamental, and sentiment analysis integration
- **Risk-Adjusted Recommendations**: Tailored advice based on risk tolerance profiles
- **Real-Time Data Integration**: Live market data, news feeds, and SEC filings

### Specialized Analysis

- **Technical Valuation**: Historical price analysis, volatility metrics, VaR calculations
- **Sentiment Analysis**: Multi-source news collection with advanced NLP processing
- **Fundamental Analysis**: SEC filing analysis, financial statement evaluation
- **Portfolio Optimization**: Risk-adjusted allocation recommendations

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Internet connection for data sources

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd alpha-agents-implementation
```

2. **Install uv (if not already installed):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Set up each agent:**

```bash
# GroupChat Agent (Coordinator)
cd agents/groupchat-agent
cp .env.example .env
# Edit .env with your OpenAI API key
uv sync
cd ../..

# Valuation Agent
cd agents/valuation-agent
cp .env.example .env
# Edit .env with your configuration
uv sync
cd ../..

# Sentiment Agent
cd agents/sentiment-agent
cp .env.example .env
# Edit .env with your configuration
uv sync
cd ../..

# Fundamental Agent
cd agents/fundamental-agent
cp .env.example .env
# Edit .env with your configuration
uv sync --native-tls
cd ../..
```

### Running the System

Start all agents in separate terminals:

```bash
# Terminal 1 - Valuation Agent
cd agents/valuation-agent
uv run python src/valuation_agent/main.py

# Terminal 2 - Sentiment Agent
cd agents/sentiment-agent
uv run python src/sentiment_agent/main.py

# Terminal 3 - Fundamental Agent
cd agents/fundamental-agent
uv run python src/fundamental_agent/main.py

# Terminal 4 - GroupChat Agent (Coordinator)
cd agents/groupchat-agent
uv run python -m groupchat_agent
```

### Testing the System

```bash
# Quick test of the GroupChat Agent
cd agents/groupchat-agent
python test_client.py

# Test individual agents
cd agents/valuation-agent
python quick_test.py

cd ../sentiment-agent
python test_client.py simple

cd ../fundamental-agent
python test_client.py
```

## Usage

### Basic Analysis Request

Send requests to the GroupChat Agent (default: `http://localhost:3000`):

```json
{
  "jsonrpc": "2.0",
  "method": "collaborative_analysis",
  "params": {
    "symbol": "AAPL",
    "query": "Should I invest in Apple stock given current market conditions?",
    "risk_tolerance": "neutral",
    "analysis_depth": "standard"
  },
  "id": "analysis_1"
}
```

### Risk Tolerance Profiles

- **Risk-Averse**: Focus on stability, low volatility, downside protection
- **Risk-Neutral**: Balanced approach considering both growth and risk
- **Risk-Seeking**: Higher tolerance for volatility in pursuit of growth

### Analysis Depth Levels

- **Quick**: Basic analysis with key metrics
- **Standard**: Comprehensive analysis with debate if needed
- **Detailed**: In-depth analysis with extended debate rounds

## API Documentation

### GroupChat Agent Endpoints

- **POST /**: Main JSON-RPC endpoint
- **GET /health**: System health check
- **GET /agent-card**: Agent capabilities

### Available Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `collaborative_analysis` | Full multi-agent analysis with debate | `symbol`, `query`, `risk_tolerance`, `analysis_depth` |
| `quick_consensus` | Fast consensus without full debate | `symbol`, `question`, `risk_tolerance` |
| `agent_health_check` | Check all specialist agents | None |
| `get_configuration` | Return current configuration | None |

### Individual Agent Endpoints

Each specialist agent exposes:

- **POST /**: A2A JSON-RPC endpoint (`message/send` method)
- **GET /health**: Health check
- **GET /agent-card**: Agent capabilities card

## Configuration

### Environment Variables

Each agent requires configuration via `.env` files. Key variables include:

```env
# Common Configuration
OPENAI_API_KEY=your_openai_api_key_here
HOST=0.0.0.0
PORT=300X  # Different for each agent
LOG_LEVEL=INFO

# Agent-Specific Configuration
AGENT_NAME=agent-name
AGENT_VERSION=1.0.0

# Analysis Parameters
DEFAULT_ANALYSIS_PERIOD_DAYS=365
DEFAULT_RISK_FREE_RATE=0.05
MAX_NEWS_ARTICLES=10
NEWS_LOOKBACK_DAYS=7
```

### Debate Configuration (GroupChat Agent)

```env
MAX_DEBATE_ROUNDS=5
CONSENSUS_THRESHOLD=0.75
DEBATE_TIMEOUT=300
COLLABORATION_MODE=debate
MIN_AGENT_RESPONSES=2
CONSOLIDATION_STRATEGY=weighted_consensus
```

## Development

### Project Structure

```
alpha-agents-implementation/
├── README.md                    # This file
├── a2a_agent_card_schema.json  # A2A protocol schema
├── agents/
│   ├── pyproject.toml          # Shared configuration
│   ├── groupchat-agent/        # Multi-agent coordinator
│   │   ├── src/groupchat_agent/
│   │   ├── tests/
│   │   └── README.md
│   ├── valuation-agent/        # Technical analysis
│   │   ├── src/valuation_agent/
│   │   ├── tests/
│   │   └── README.md
│   ├── sentiment-agent/        # News sentiment analysis
│   │   ├── src/sentiment_agent/
│   │   ├── tests/
│   │   └── README.md
│   └── fundamental-agent/      # Financial statement analysis
│       ├── src/fundamental_agent/
│       ├── tests/
│       └── README.md
```

### Testing

Run tests for each agent:

```bash
# All agents
find agents -name "test_*.py" -exec python {} \;

# Individual agent tests
cd agents/groupchat-agent && uv run pytest tests/
cd agents/valuation-agent && uv run pytest tests/
cd agents/sentiment-agent && uv run pytest tests/
cd agents/fundamental-agent && uv run pytest tests/
```

### Code Quality

```bash
# Format code
find agents -name "*.py" -path "*/src/*" -exec uv run black {} +

# Lint code
find agents -name "*.py" -path "*/src/*" -exec uv run ruff check {} +

# Type checking
cd agents/valuation-agent && uv run mypy src/
# Repeat for other agents
```

## Multi-Agent Collaboration

### Debate Mechanism

1. **Initial Coordination**: All specialist agents analyze the query simultaneously
2. **Consensus Analysis**: Agreement levels calculated across agent responses
3. **Debate Rounds**: Structured arguments when consensus < threshold
4. **Evidence Exchange**: Agents present counter-arguments and supporting data
5. **Convergence**: Re-evaluate consensus after each debate round
6. **Consolidation**: Generate weighted final recommendation

### Consensus Scoring

- **Technical Agreement**: Valuation metrics alignment
- **Sentiment Consistency**: News sentiment correlation
- **Fundamental Coherence**: Financial analysis agreement
- **Overall Consensus**: Weighted average across all dimensions

## Data Sources

### Market Data

- **Yahoo Finance**: Historical OHLCV data, company information
- **Real-time Feeds**: Live market data integration

### News Sources

- **Yahoo Finance News**: Financial news and earnings coverage
- **Google News**: Broad market coverage and breaking news
- **RSS Feeds**: Real-time financial news aggregation

### Fundamental Data

- **SEC EDGAR**: 10-K and 10-Q filing analysis
- **Financial Statements**: Balance sheet, income statement, cash flow analysis
- **Industry Data**: Sector comparison and benchmarking

## Performance Optimization

### Caching Strategy

- **Historical Data**: Yahoo Finance data cached per session
- **News Articles**: Recent sentiment analysis cached temporarily
- **Agent Responses**: Conversation history windowed to prevent memory leaks

### Concurrency

- **Async Operations**: All agent communications are asynchronous
- **Parallel Requests**: Specialist agents queried simultaneously
- **Connection Pooling**: Efficient HTTP client connection management

### Rate Limiting

- **Yahoo Finance**: Respects API rate limits
- **News Sources**: Implements backoff strategies
- **OpenAI API**: Request throttling and retry mechanisms

## Monitoring and Logging

### Logging Levels

- **DEBUG**: Detailed debugging information and request/response logging
- **INFO**: General operational information and agent coordination
- **WARNING**: Non-critical issues and fallback scenarios
- **ERROR**: Critical errors and system failures

### Health Monitoring

- **Agent Health**: Individual agent availability and performance
- **Data Source Status**: External API availability and response times
- **System Metrics**: Memory usage, response times, error rates

### Log Files

- Each agent logs to its own file (e.g., `valuation-agent.log`)
- Centralized logging available through GroupChat Agent
- Configurable log rotation and retention

## Security Considerations

### API Key Management

- Store API keys in `.env` files (not version controlled)
- Use environment-specific configurations
- Implement key rotation strategies

### Data Privacy

- No sensitive financial data is stored permanently
- Conversation history is session-scoped
- All external API communications use HTTPS

### Access Control

- Agents communicate over localhost by default
- Configurable host binding for production deployment
- No authentication required for localhost development

## Troubleshooting

### Common Issues

**Agent Connection Errors:**

- Verify all agents are running on correct ports
- Check `.env` configuration files
- Ensure no port conflicts with other services

**OpenAI API Issues:**

- Verify API key is valid and has sufficient credits
- Check rate limiting and quota usage
- Ensure network connectivity to OpenAI services

**Data Source Failures:**

- Yahoo Finance may have temporary outages
- News sources may implement rate limiting
- SEC EDGAR has maintenance windows

**Consensus Timeout:**

- Increase `DEBATE_TIMEOUT` in GroupChat Agent config
- Reduce `MAX_DEBATE_ROUNDS` for faster responses
- Adjust `CONSENSUS_THRESHOLD` for easier agreement


## Changelog

### Version 1.0.0

- Initial release with four specialized agents
- A2A protocol compliance across all agents
- Multi-agent debate mechanism implementation
- Comprehensive testing suite
- Documentation and examples