# Recall AI Trading Agent

Enterprise-grade modular trading bot for Recall Network AI agent competitions.

## Architecture

**Production-ready modular design:**
- `main.py` - Main orchestrator with comprehensive error handling
- `config.py` - Environment-based configuration management  
- `data_handler.py` - Enterprise API client with retry logic
- `strategy_engine.py` - Extensible trading strategy framework
- `executor.py` - Trade execution engine with performance tracking
- `utils.py` - Structured logging and utilities

**Legacy versions:**
- `recall_trading_agent.py` - Single-file implementation
- `recall_trading_agent_v1_working.py` - Backup version

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Run
```bash
python main.py
```

## Features

- Enterprise Architecture - Modular, maintainable, extensible
- Production Ready - Comprehensive error handling & logging
- Risk Management - Position sizing, stop-loss, take-profit
- Performance Tracking - Detailed metrics and trade history
- Multi-Environment - Sandbox and production support
- Strategy Engine - Pluggable trading strategies
- API Integration - Robust Recall Network API client

## Performance

**Demo Results:**
- Success Rate: 100%
- Total Trades: 4 successful executions
- Environment: Sandbox verified
- Latency: <500ms average execution time

## Security

- API keys stored in environment variables
- No hardcoded credentials in source code
- .env file excluded from version control

## Development

Built with modern Python practices:
- Type hints and dataclasses
- Structured logging with context
- Retry logic with exponential backoff
- Comprehensive error handling
- Performance monitoring