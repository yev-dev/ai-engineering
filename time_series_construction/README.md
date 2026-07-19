
# Time Series Construction

## Quick Start

Use the project's conda environment before running the application:

```bash
cd /Users/yevyerm/Dev/Projects/ai/ai-engineering
conda activate ai_engineering
python -m time_series_v2.cli
```

The default model is read from `LLM_MODEL` and falls back to
`ollama/gemma4:26b`. For a different LiteLLM-compatible provider, configure
the provider's environment variables and set `LLM_MODEL` before starting the
CLI. Generated artifacts are written to `~/time_series_construction` unless
`TIME_SERIES_OUTPUT_DIR` is set.

For a direct module check without starting an LLM session:

```bash
conda activate ai_engineering
python -c "import time_series_v2; print('time_series_v2: ok')"
```

### Logging

The CLI logs workflow steps for requests, agent iterations and actions, LLM
calls, tool execution, callback events, and human-input pause/resume handling.
Logs do not include full prompts, transcripts, or price arrays at INFO level.

```bash
LOG_LEVEL=INFO python -m time_series_v2.cli
LOG_LEVEL=DEBUG LOG_FILE=~/time_series_construction/application.log python -m time_series_v2.cli
```

`LOG_LEVEL` defaults to `INFO`. Set `LOG_FILE` to add a persistent file sink;
the parent directory is created automatically. The logging setup is applied
by the CLI and can also be called directly with `configure_logging()` from
`logging_config.py` for another application entry point.

### Workflow Tests

Install the test runner in `ai_engineering` and run the deterministic workflow
tests without an LLM:

```bash
conda activate ai_engineering
python -m pip install pytest
python -m pytest time_series_v2/tests/test_workflow.py -q
```

The tests use the bundled fixture data and verify successful lookup, quality
metrics, gap filling, delegation, and clear handling of unavailable data.

This project helps to create financial continues time series from multiple data sources like Reuters, Bloomberg and Yahoo. Orchestration and Agents should replicate
overral architecture presented in callbacks folder. Langchain framework should be used to manage ReACT-style orchestrations. human in a loop input expected to process.

The following architure should be adopted based on implmentation in callbacks folder

All agents should have ReACT architecture. A. agent definition is Agent with name, description, system_prompt, tools and optional gard_rails. We have the following agents:
Orchestrator - initial stage analyses request and creates a workflow. The workflow should not be hard coded but LLM driven
ReferenceData agent - enriches provided financial asset information. All assets information is taken from data/instruments.csv file to resolve. An aggent should understand that symbol can be mapped ticker, security name can be also asset name, 
MarketData agent - get historical market data for a given date range and ticker name from available data sources
DataQuality agent - check provided time series for financial time series data quality issues. This agent does check for missing values, number of NaNs, and other commonly accepted data quality checks for financial to come back with report for each data source for a given asset
GapFilling agent - this agent is activated after user selected preferred data source. it offers to chose from available financial time series gap filling methodologies.  
TimeSeriesConstruction agent - give time series upload time series to target system. At the moment this will be generated csv file. 
ReportingAgent - expects serialised data and presents rich command line nice table with summary 

The architure should be flexiable to accomodate  new agents in the future. Agents definition should be kepts in a agents_definition.py file for maintenance purpose

We want to use langain tools annotation to develop tools for agents. 

Similiar to callback folder we want to use litellm library for ollama and github copilot clients via factory. Reuse existed architecture. 

Handler - the system should utialise callback architecture. Create and register TimeSeriesConstructionHander that extends from langchain BaseCallbackHandler to handle all llm callbacks. Intorduce CallbackEvent with CallbackEventType that fits our business case. We should be able to handle user response to pause and have a logic to handle CallbackEventType with error handling mechanism. Other commonly used llm handlers should be 

The system should accomodate more handlers in the future with different use cases so the architecture should be flexible with processor to take in a list of availbale handlers

For each run we can generate multiple outputs. System creates time_series_construction folder in user's directory if it doesn't exist and per each run there will be RUN ID folder with date. Inside of that folder each agent can save outputs identified by agent

The implementations includes human-in-the-loop checkpoints with the following business flow:

1. User is asked what he wants to do. At the initial stage user provides a ticker/symbol or instrument name/ticker name/asset name, start/end date in any format either in wording or numbers format handled by Orchestrator agent

3. The system retrieves instrument details by an identifier with a help from ReferenceData informs user about it and progresses further to MarketData agent to retrieve historical prices for all available data sources. Available data sources are yahoo, bloomberg, reuters provided by available_data_sources tool.

3. MarketData agent pulls historical prices from multi-source connectors:
  - Yahoo
  - Bloomberg
  - Reuters 
  data is provided by historical_prices tool that uses stock data service to load corresponding csv files from data directory for 
4. Data quality agent compares each source using financial data quality metrics. This is being fed into ReportingAgent that present a summary and corresponding event activates human in a loop request waiting for user to choose a data source by name or number. A user has an option to download time series for each data source that is by default saved into users' home directory into 'time_series_construction' folder that is created if it doesn't exist. Proposed time series is downloaded into a single csv file with a name of agent and date and time suffix. It has dates as index and columns name of data sources. We also populate report csv with statistics summary for data source check as csv file. This is being handled by ReportAgent with corresponding tools. We also what to create seaborn graph to visualise time series prices, visualise any data quality issues to compare. This should be handled by correspondig tool 
5. After data source is selected a user presented with a option choose from available data sources or exit or start again 
6. If user wishes to continue Orchestrator agent progresses it further GapFilling agent
5. Gap-filling analysis agent recommends methods based on data. We can have simple implememtation for now that only does different interperlation flavours but this will be extended in the future
6. Use chooses a method and progresses further
7. System applies GapFilling methodology and wait for user to reply with the next step. This can be continue to Time Series construction, Download data
8. User presented with option to continue or to generate output of time series based on applied method. Output with csv that has index date and columns per each method and selected method presented. We also generate graphs that shows each method results and original to compare on the same seaborn chart 
8. Time-series generation agent builds a continuous series and generate fianl time series file into time_series_construction folder
9. Reporting agent prints summary and full ReACT trace that is also saved. We want to save in text file all LLM reasoning outputs for future valuation and comparasion and reinforcement learning training. That is being exposrted to folder as artifact with id


