# News Discovery Agent

A self-directed news discovery agent which starts with a root keyword, queries NewsAPI for recent articles and uses a locally hosted LLM to analyze each batch of results.

For every keyword searched, the LLM decides whether the topic is emerging or "hot", suggests up to three related follow-up keywords to explore next and flags specific articles from the current batch as "super hot" based on importance. The agent expands its search queue using those suggested keywords until it reaches a daily quota, logging every search, LLM decision and article.

At the end of the run, it produces three JSON files:

1. Full execution log

2. Deduplicated timeline of only the "super hot" stories

3. Deduplicated timeline of all collected stories

Effectively creating an automated, LLM-guided news trend exploration and prioritization pipeline.
