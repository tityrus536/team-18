# QDC 2026 – ReBel AgenticAI Challenge

Code for our team's AI agents in the Belfius x QTEM Data Challenge.
Documents, logs and reports live in the shared Google Drive; code lives here.

## How it works

```
Data -> Technical + Macro (in parallel) -> Portfolio Options -> Risk -> Challenger -> humans decide
                                              ^                  |
                                              +--- if "fail" ----+
```

- **Python calculates, the LLM interprets.** Indicators are plain Python functions
  (`qdc/indicators.py`). Agents call them as tools and explain the results.
- **All agents share one state** (`qdc/agents/state.py`). Each agent reads it and
  returns only its own part. Agree any change to this file with the team.
- **No look-ahead.** Agents read saved prices cut at an `as_of` date, never live data.
- **Few LLM requests.** The free Gemini tier allows about 20 requests a day per model.
  So Python gathers and calculates everything first, and each agent sends it to the
  LLM in one or a few grouped requests (Data agent: 1 per 10 ETFs; Technical: 1 per 25).
- **Humans decide.** Agents give views and evidence; the team approves every order
  and logs it in the Decision Log (Drive, folder 03).

## Status

| Agent | File | Status | Owner |
|---|---|---|---|
| Data | `qdc/agents/data_agent.py` | **Working**: finds tickers (AI) + quality checks | |
| Technical | `qdc/agents/technical.py` | **Working first version** | |
| Macro/News | placeholder | To build | |
| Portfolio Options | placeholder | To build | |
| Risk | placeholder | To build | |
| Challenger | placeholder | To build | |

## Setup (once, about 15 minutes)

1. Install **Python 3.11 or newer** from python.org, and **VS Code** with the
   "Python" and "Jupyter" extensions.
2. Clone the repo and open the folder in VS Code.
3. In the VS Code terminal, create the environment and install everything:
   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # Mac/Linux:    source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Then `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.
4. Get a free Gemini API key at aistudio.google.com. Copy `.env.example` to `.env`
   and paste your key. **Never commit `.env`.** (In China, you need a VPN for Gemini.)
5. Check everything works (no key or internet needed):
   ```bash
   pytest
   ```

## Daily use

Run everything from the project root:

```bash
python scripts/update_prices.py                       # download prices into data/
python scripts/run_technical_agent.py                 # Technical agent on the latest data
python scripts/run_technical_agent.py --as-of 2026-10-30   # as if it were that date
python scripts/run_pipeline.py                        # whole workflow, placeholders only
python scripts/run_pipeline.py --real data technical  # whole workflow, real Data and Technical agents
```

Reports are saved in `outputs/`. Copy the ones you use to the Drive.

## Adding ETFs: the Data agent

The Data agent finds Yahoo tickers from the ISINs, and a human approves them:

```bash
python scripts/find_tickers.py --limit 3   # test on 3 ETFs first
python scripts/find_tickers.py             # all ETFs without a ticker
```
1. Open `outputs/ticker_proposals.csv` and check each proposal (name, currency,
   confidence, reason). Verify the low and medium ones on justETF.
2. Write `yes` in the `approved` column for the ones you accept.
3. `python scripts/apply_tickers.py` copies them into `config/etf_universe.csv`.
4. `python scripts/update_prices.py` then `python scripts/check_data.py`
   for the quality report (short histories, gaps, stale prices, unusual moves).

You can also fill a ticker by hand in `config/etf_universe.csv`
(currency is `EUR`, `USD`, `GBP` or `GBp` for London prices in pence).

## Building your agent

1. Copy the structure of `qdc/agents/technical.py`: tools (Python functions with
   `@tool`), a system prompt, and an output format (a Pydantic model).
2. Write a node function `def my_node(state) -> dict` that returns only your key
   of the shared state (for example `{"macro_view": ...}`).
3. Replace your placeholder in `AGENTS` in `qdc/agents/graph.py`.
4. Add a test in `tests/` and run `pytest` before opening a pull request.

## Working together on GitHub

- Never work on `main` directly. Create a branch: `git checkout -b macro-agent`.
- Commit small steps with clear messages, push, and open a pull request.
- Another member reviews and merges. `pytest` must pass first.
- Pull `main` every morning: `git pull origin main`.

## Project structure

```
config/etf_universe.csv   the 50 ETFs from the Belfius deck (27 Oct), tickers filled so far
qdc/data.py               download, EUR conversion, load prices cut at as_of
qdc/indicators.py         returns, volatility, drawdown, trend, momentum ranking
qdc/quality.py            data quality checks
qdc/llm.py                the model the whole team uses
qdc/agents/state.py       the shared state
qdc/agents/data_agent.py  the Data agent
qdc/agents/technical.py   the Technical agent
qdc/agents/placeholders.py fake agents until the real ones exist
qdc/agents/graph.py       all agents connected with LangGraph
scripts/                  commands to run
tests/                    checks that run without an API key
notebooks/                exploration (keep final code in qdc/)
```
