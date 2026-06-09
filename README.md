# LLM News Discovery Agent

Self-directed news discovery agent which starts from a root keyword, searches `NewsAPI` for recent articles, uses a local Ollama LLM to evaluate stories and saves structured output files for later review or visualization.

## Application Overview

Starts from a configured root keyword and searches `NewsAPI` for recent articles. Each batch of results is normalized into a consistent format, then sent to a local Ollama model for analysis. The model evaluates whether the keyword is producing interesting, emerging or high-impact news, explains its reasoning, suggests 3 related follow-up keywords and identifies articles from the batch which are "super hot".

The agent uses the LLM’s suggested keywords to expand its search queue while avoiding duplicate searches and stopping when the daily quota is reached. Throughout the run, it logs every keyword search, returned article batch, LLM decision and flagged super-hot article. At the end it saves a full run log, deduplicated JSON timeline of super-hot stories, deduplicated JSON timeline of all discovered stories and plain-text export of all discovered story content.

## Basic Setup Instructions

Below are the required software programs and instructions for installing and using this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

- [Ollama](https://ollama.com/)

### Steps

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/llm-news-discovery-agent.git`

4. Navigate to the repo's directory: `cd llm-news-discovery-agent`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate the virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Create a `.env` file from the included template: `cp .env.template .env`

9. Open and configure the `.env` file: `nano .env`

10. Pull a local LLM using Ollama, for example: `ollama pull mistral`

11. Confirm the model is available: `ollama run mistral`

12. Run the main script: `python3 app.py`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Search for recent articles based on a starting keyword and analyze the results with a locally running Ollama LLM

- Identify high-impact news topics, suggest related follow-up keywords and automatically expand the search queue

- Flag especially important articles as "super hot" stories and save them into a deduplicated JSON file

- Generate structured output files including super-hot stories and a plain-text export of article content

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
