# Sentiment Agent

A specialized A2A-compliant agent for financial news sentiment analysis. This agent is part of the AlphaAgents portfolio analysis system and provides comprehensive sentiment analysis using multi-source news collection and advanced NLP techniques.

## Features

- **Multi-Source News Collection**: Gathers news from Yahoo Finance, Google News, and financial RSS feeds
- **Advanced Sentiment Analysis**: Uses both TextBlob and VADER sentiment analysis for robust results
- **Investment Implications**: Provides actionable investment recommendations based on sentiment analysis
- **Risk-Adjusted Analysis**: Tailored recommendations based on risk tolerance profiles
- **A2A Protocol Compliant**: Full JSON-RPC 2.0 implementation with streaming support

## Installation

This agent uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the agent directory
cd agents/sentiment-agent

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
PORT=3002

# Agent Configuration
AGENT_NAME=sentiment-agent
AGENT_VERSION=1.0.0
LOG_LEVEL=INFO

# News Collection Configuration
MAX_NEWS_ARTICLES=10
NEWS_LOOKBACK_DAYS=7
USER_AGENT=Mozilla/5.0 (compatible; SentimentAgent/1.0)

# Sentiment Analysis Configuration
SENTIMENT_THRESHOLD_POSITIVE=0.1
SENTIMENT_THRESHOLD_NEGATIVE=-0.1
```

## Usage

### Starting the Agent Server

```bash
# Run with uv
uv run python src/sentiment_agent/main.py

# Or activate the virtual environment first
uv shell
python src/sentiment_agent/main.py
```

The agent will start an A2A-compliant JSON-RPC server on the configured host and port (default: `http://localhost:3002`).

### Testing the Agent

Several test scripts are provided to verify the agent is working correctly:

#### 1. Quick Python Test
```bash
python test_client.py simple
```
Simple test that checks health and performs one Apple sentiment analysis.

#### 2. Comprehensive Python Test
```bash
python test_client.py          # Full test suite
```
Advanced test client with multiple test scenarios including risk tolerance variations.

#### 3. Basic Tools Test
```bash
python test_basic.py
```
Tests the underlying tools (news collection, sentiment analysis) without the full agent.

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
          "text": "Analyze Apple's recent news sentiment and market perception"
        }
      ]
    },
    "metadata": {
      "risk_tolerance": "neutral",
      "lookback_days": 7,
      "max_articles": 10
    }
  }
}
```

## Capabilities

### Skills

1. **News Sentiment Analysis**
   - Multi-source news collection and analysis
   - TextBlob and VADER sentiment scoring
   - Investment implication assessment

2. **Analyst Ratings Analysis**
   - Rating changes and coverage analysis
   - Market impact assessment
   - Consensus sentiment evaluation

3. **Market Event Sentiment**
   - Corporate event sentiment analysis
   - Earnings announcement impact
   - Executive changes and insider trading analysis

4. **Sentiment Risk Assessment**
   - Risk-adjusted sentiment analysis
   - Market mood evaluation
   - Volatility prediction based on news sentiment

### Tools

The agent has access to these specialized tools:

- **Company Name Resolver**: Converts company names to stock tickers
- **Stock News Collection**: Multi-source news gathering with deduplication
- **News Summarization**: Advanced sentiment analysis with investment implications

### Risk Tolerance Profiles

- **Risk-Averse**: Focus on stable sentiment, avoid high volatility from negative news
- **Risk-Neutral**: Balance news sentiment with broader market context
- **Risk-Seeking**: May see negative sentiment as buying opportunities

### News Sources

- **Yahoo Finance**: Financial news and earnings coverage
- **Google News**: Broad market coverage and breaking news
- **RSS Feeds**: Real-time financial news feeds

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
sentiment-agent/
├── src/sentiment_agent/
│   ├── __init__.py
│   ├── agent.py          # Langchain agent implementation
│   ├── tools.py          # News collection and sentiment analysis tools
│   ├── server.py         # A2A JSON-RPC server
│   └── main.py           # Entry point
├── tests/                # Test suite
├── pyproject.toml        # uv configuration
├── .env.example          # Environment template
└── README.md
```

## Integration

This agent is designed to work as part of the AlphaAgents multi-agent system:

1. **Standalone Operation**: Can be used independently for sentiment analysis
2. **GroupChat Integration**: Works with the GroupChat agent for multi-agent collaboration
3. **Portfolio System**: Integrates with other agents (Fundamental, Valuation) for comprehensive analysis

## Sentiment Analysis Methodology

### TextBlob Analysis
- Academic approach to sentiment analysis
- Polarity score from -1 (negative) to +1 (positive)
- Subjectivity score from 0 (objective) to 1 (subjective)

### VADER Analysis
- Optimized for social media and news text
- Compound score from -1 to +1
- Individual positive, negative, and neutral scores

### Combined Scoring
- Weighted combination of TextBlob (40%) and VADER (60%)
- Confidence levels based on score magnitude
- Investment recommendations with risk considerations

## Logging

The agent logs to both console and file (`sentiment-agent.log`). Log levels can be configured via the `LOG_LEVEL` environment variable.

## Error Handling

The agent includes comprehensive error handling:
- Invalid ticker symbols are caught and reported
- News collection failures are gracefully handled
- A2A protocol compliance is maintained even during errors
- Rate limiting and timeout protection for news sources
- All errors are logged for debugging

## Performance

- **Asynchronous Operations**: All operations are asynchronous for better performance
- **News Caching**: Recent news collection results are cached temporarily
- **Request Deduplication**: Duplicate news articles are filtered out
- **Rate Limiting**: Respects news source API rate limits
- **Memory Management**: Conversation history is windowed to prevent memory leaks

## Limitations

- News availability depends on external sources
- Some news sources may have rate limits
- Sentiment analysis accuracy varies with text quality
- Historical news coverage may be limited for some stocks
- Real-time sentiment analysis depends on news publication timing

## Support

For issues or questions:
1. Check the logs in `sentiment-agent.log`
2. Verify your `.env` configuration
3. Ensure all dependencies are installed with `uv sync`
4. Check that the OpenAI API key is valid and has sufficient credits
5. Verify internet connectivity for news collection