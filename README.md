# LLM News Discovery Agent

A self-directed news discovery agent which starts with a root keyword, queries NewsAPI for recent articles and uses a locally hosted LLM to analyze each batch of results.

## Overview

For every keyword searched, the LLM decides whether the topic is emerging or "hot", suggests up to three related follow-up keywords to explore next and flags specific articles from the current batch as "super hot" based on importance. The agent expands its search queue using those suggested keywords until it reaches a daily quota, logging every search, LLM decision and article.

At the end of the run, it produces three JSON files:

1. Full execution log

2. Deduplicated timeline of only the "super hot" stories

3. Deduplicated timeline of all collected stories

Effectively creating an automated, LLM-guided news trend exploration and prioritization pipeline.

## Set Up

By cloning this repo it is assumed you are already familiar with the hardware and software technologies needed to install and use the application. Along these lines, below are the required software programs and instructions for installing and using this application.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

- [Ollama](https://ollama.com/)

### Steps

1. Install the above programs

2. Open a terminal

3. Clone this repository using `git` by running the following command: `git clone git@github.com:devbret/news-discovery-agent.git`

4. Navigate to the repo's directory by running: `cd news-discovery-agent`

5. Create a virtual environment by running this command in the terminal: `python3 -m venv venv`

6. Activate the virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Create and configure an `.env` file at the root of this project

9. Locally install an LLM using Ollama, for example: `ollama pull mistral`

10. Ensure your LLM is available by running a command such as: `ollama run mistral`

11. Run the main script with the command: `python3 app.py`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Automatically discover and expand emerging news topics by combining NewsAPI data with LLM exploration

- Identify and prioritize the most important stories while generating structured outputs for analysis and visualization

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
