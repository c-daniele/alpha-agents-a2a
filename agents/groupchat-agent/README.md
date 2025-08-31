# GroupChat Agent

The GroupChat Agent serves as the entry point for end-users and coordinates the three specialist agents (Valuation, Sentiment, and Fundamental) using a structured debate approach to generate collaborative stock analysis reports.

## Features

- **Multi-Agent Coordination**: Orchestrates communication between three specialist agents
- **Structured Debate Mechanism**: Implements Round Robin debate approach with consensus thresholds
- **Collaborative Analysis**: Consolidates diverse perspectives into weighted recommendations  
- **A2A JSON-RPC 2.0 Server**: Standard agent-to-agent communication protocol
- **Configurable Debate Parameters**: Customizable rounds, consensus thresholds, and timeouts

## Architecture

The GroupChat Agent implements a sophisticated multi-agent coordination system:

1. **Initial Coordination**: Gathers analysis from all specialist agents simultaneously
2. **Consensus Analysis**: Evaluates agreement levels and identifies areas of disagreement  
3. **Debate Rounds**: Facilitates structured arguments when consensus isn't reached
4. **Consolidation**: Generates weighted final recommendations based on all inputs

## Configuration

Environment variables (`.env`):

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=3000

# Agent Configuration  
AGENT_NAME=groupchat-agent
AGENT_VERSION=1.0.0
LOG_LEVEL=INFO

# Agent Registry Configuration
AGENT_REGISTRY_URL=http://localhost:8000

# Specialist Agent Names (for registry lookup)
VALUATION_AGENT_NAME=valuation
SENTIMENT_AGENT_NAME=sentiment
FUNDAMENTAL_AGENT_NAME=fundamental

# Debate Configuration
MAX_DEBATE_ROUNDS=5
CONSENSUS_THRESHOLD=0.75
DEBATE_TIMEOUT=300
COLLABORATION_MODE=debate

# Analysis Configuration
MIN_AGENT_RESPONSES=2
CONSOLIDATION_STRATEGY=weighted_consensus
ENABLE_DISCUSSION_LOGS=true
```

## Installation

```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

## Usage

### Starting the Server

```bash
# Start the GroupChat Agent server
python -m groupchat_agent

# Or using uv
uv run python -m groupchat_agent
```

### Running Tests

```bash
# Run test suite
python -m groupchat_agent --mode test

# Or run tests directly
python -c "from groupchat_agent import run_tests; run_tests()"
```

## API Methods

### `collaborative_analysis`

Conducts comprehensive multi-agent analysis using debate approach.

**Parameters:**
- `symbol` (string): Stock symbol to analyze
- `query` (string): Analysis question or request
- `risk_tolerance` (string): "averse", "neutral", or "seeking" (default: "neutral")
- `analysis_depth` (string): "quick", "standard", or "detailed" (default: "standard")

**Example:**
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

### `quick_consensus`

Performs quick consensus check across specialist agents without full debate.

**Parameters:**
- `symbol` (string): Stock symbol to analyze
- `question` (string): Specific question to ask all agents
- `risk_tolerance` (string): Risk tolerance level (default: "neutral")

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "quick_consensus",
  "params": {
    "symbol": "TSLA", 
    "question": "What is the current sentiment for Tesla?",
    "risk_tolerance": "seeking"
  },
  "id": "consensus_1"
}
```

### `agent_health_check`

Checks the health status of all specialist agents.

**Parameters:** None

### `get_configuration`

Returns current agent configuration and settings.

**Parameters:** None

## Testing

The GroupChat Agent includes comprehensive testing:

```python
from groupchat_agent import GroupChatTestClient, run_tests

# Run all tests
results = run_tests()

# Use test client directly
client = GroupChatTestClient("http://localhost:3000")
health = await client.test_health_check()
```

## Agent Registry Integration

The GroupChat Agent uses dynamic agent discovery through an Agent Registry:

### Registry Configuration
- **Registry URL**: `http://localhost:8000` (configurable via `AGENT_REGISTRY_URL`)
- **API Endpoint**: `GET /agents?name={agent_name}` 
- **Search Method**: LIKE-based search for agent names

### Specialist Agents Discovery
The GroupChat Agent automatically discovers specialist agents by name:
1. **Valuation Agent**: Searches for `"valuation"` in registry
2. **Sentiment Agent**: Searches for `"sentiment"` in registry  
3. **Fundamental Agent**: Searches for `"fundamental"` in registry

### Runtime Behavior
- Agent URLs are fetched dynamically on first analysis request
- 5-minute cache for agent URLs to reduce registry calls
- Graceful error handling if registry or agents are unavailable
- Health checks include registry connectivity status

### Requirements
- Agent Registry must be running at configured URL
- Specialist agents must be registered with correct names
- GroupChat Agent will fail gracefully if agents are not found

## Multi-Agent Debate Process

1. **Coordination Phase**: Query all specialist agents simultaneously
2. **Consensus Analysis**: Calculate agreement levels and identify conflicts
3. **Debate Rounds**: If consensus < 75%, initiate structured debates
4. **Evidence Exchange**: Agents present counter-arguments and supporting data
5. **Convergence Check**: Re-evaluate consensus after each round
6. **Consolidation**: Generate weighted final recommendation

## Logging and Monitoring

The agent provides comprehensive logging:
- Agent coordination activities
- Debate round progression  
- Consensus analysis results
- Error handling and recovery
- Performance metrics

Set `LOG_LEVEL=DEBUG` for detailed debugging information.

## Error Handling

The agent handles various error scenarios:
- Specialist agent unavailability  
- Network timeouts and connection issues
- Conflicting analysis results
- Debate round failures
- Consensus timeout scenarios

All errors are logged and returned in JSON-RPC error format.

## Development

### Project Structure

```
groupchat-agent/
├── src/groupchat_agent/
│   ├── __init__.py          # Package initialization
│   ├── agent.py             # Main GroupChat Agent class
│   ├── server.py            # A2A JSON-RPC server
│   ├── tools.py             # Coordination and debate tools
│   ├── registry_service.py  # Agent Registry integration
│   ├── test_client.py       # Test client and utilities
│   └── __main__.py          # CLI entry point
├── tests/                   # Test files
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── .env                    # Environment variables
└── .env.example            # Environment template
```

### Contributing

1. Follow existing code style and patterns
2. Add comprehensive tests for new features  
3. Update documentation for API changes
4. Ensure all specialist agent integrations work correctly
5. Test debate mechanisms thoroughly