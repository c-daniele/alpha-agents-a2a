# Valuation Agent

A specialized A2A-compliant agent for stock valuation and technical analysis. This agent is part of the AlphaAgents portfolio analysis system and provides comprehensive valuation analysis using historical price and volume data.

## Features

- **Technical Analysis**: Historical price and volume analysis with comprehensive metrics
- **Volatility & Risk Metrics**: Calculates volatility, Sharpe ratios, VaR, maximum drawdown
- **Company Name Resolution**: Converts company names and ISIN codes to stock tickers
- **Risk-Adjusted Analysis**: Tailored recommendations based on risk tolerance profiles
- **A2A Protocol Compliant**: Full JSON-RPC 2.0 implementation with streaming support

## Installation

This agent uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the agent directory
cd agents/valuation-agent

# Install dependencies
uv sync
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

# Server Configuration  
HOST=0.0.0.0
PORT=3001

# Agent Configuration
AGENT_NAME=valuation-agent
AGENT_VERSION=1.0.0
LOG_LEVEL=INFO

# Analysis Configuration
DEFAULT_ANALYSIS_PERIOD_DAYS=365
DEFAULT_RISK_FREE_RATE=0.05
TRADING_DAYS_PER_YEAR=252
```

## Usage

### Starting the Agent Server

```bash
# Run with uv
uv run python src/valuation_agent/main.py

# Or activate the virtual environment first
uv shell
python src/valuation_agent/main.py
```

The agent will start an A2A-compliant JSON-RPC server on the configured host and port (default: `http://localhost:3001`).

### Testing the Agent

Several test scripts are provided to verify the agent is working correctly:

#### 1. Quick Python Test
```bash
python quick_test.py
```
Simple test that checks health and performs one Apple stock analysis.

#### 2. Comprehensive Python Test
```bash
python test_client.py          # Full test suite
python test_client.py simple   # Simple version
```
Advanced test client with multiple test scenarios and detailed output.

#### 3. Curl-based Test
```bash
./test_with_curl.sh
```
Shell script using curl commands to test the agent endpoints.

#### 4. Basic Tools Test
```bash
python test_basic.py
```
Tests the underlying tools (Yahoo Finance, calculations) without the full agent.

### API Endpoints

- **POST /**: Main JSON-RPC endpoint for A2A protocol
- **GET /health**: Health check endpoint
- **GET /agent-card**: Returns the agent's capabilities card

### Example A2A Request

```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "id": "req-1",
  "params": {
    "message": {
      "kind": "message",
      "messageId": "msg-1",
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Analyze Apple stock with comprehensive valuation metrics"
        }
      ]
    },
    "metadata": {
      "risk_tolerance": "neutral",
      "analysis_period_days": 365
    }
  }
}
```

## Capabilities

### Skills

1. **Technical Valuation Analysis**
   - Historical price and volume analysis
   - Annualized returns and volatility calculation
   - Trend analysis and market expectations assessment

2. **Volatility and Risk Metrics**
   - Daily and annualized volatility
   - Value at Risk (VaR) calculations
   - Maximum drawdown analysis
   - Sharpe ratio and risk-adjusted returns

3. **Comparative Valuation**
   - Sector benchmarking
   - Peer comparison analysis
   - Relative performance evaluation

4. **Portfolio Allocation Insights**
   - Optimal allocation recommendations
   - Timing analysis based on technical indicators
   - Risk tolerance consideration

### Tools

The agent has access to these specialized tools:

- **Company Name Resolver**: Converts company names/ISIN to stock tickers
- **Yahoo Finance Data Fetcher**: Downloads historical OHLCV data
- **Volatility Calculator**: Computes comprehensive risk and return metrics

### Risk Tolerance Profiles

- **Risk-Averse**: Focus on stability, low volatility, limited downside risk
- **Risk-Neutral**: Balance growth potential with reasonable risk levels
- **Risk-Seeking**: Emphasis on growth potential, higher volatility tolerance

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black src/
uv run ruff check src/
```

### Type Checking

```bash
uv run mypy src/
```

## Architecture

```
valuation-agent/
├── src/valuation_agent/
│   ├── __init__.py
│   ├── agent.py          # Langchain agent implementation
│   ├── tools.py          # Yahoo Finance and calculation tools
│   ├── server.py         # A2A JSON-RPC server
│   └── main.py           # Entry point
├── tests/                # Test suite
├── pyproject.toml        # uv configuration
├── .env.example          # Environment template
└── README.md
```

## Integration

This agent is designed to work as part of the AlphaAgents multi-agent system:

1. **Standalone Operation**: Can be used independently for valuation analysis
2. **GroupChat Integration**: Works with the GroupChat agent for multi-agent collaboration
3. **Portfolio System**: Integrates with other agents (Fundamental, Sentiment) for comprehensive analysis

## Logging

The agent logs to both console and file (`valuation-agent.log`). Log levels can be configured via the `LOG_LEVEL` environment variable.

## Error Handling

The agent includes comprehensive error handling:
- Invalid ticker symbols are caught and reported
- Data fetching errors are gracefully handled
- A2A protocol compliance is maintained even during errors
- All errors are logged for debugging

## Performance

- **Data Caching**: Historical data is fetched on-demand
- **Async Operations**: All operations are asynchronous for better performance
- **Memory Management**: Conversation history is windowed to prevent memory leaks
- **Rate Limiting**: Respects Yahoo Finance API rate limits

## Support

For issues or questions:
1. Check the logs in `valuation-agent.log`
2. Verify your `.env` configuration
3. Ensure all dependencies are installed with `uv sync`
4. Check that the OpenAI API key is valid and has sufficient credits