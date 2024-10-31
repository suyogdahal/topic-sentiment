# Social Media Sentiment Analysis for Political Trends

This project was developed as part of the CSGS Hackathon, Project M2: Predict Global Trends from Social Media.

## Project Overview

This project analyzes Twitter data to uncover sentiment trends and topic preferences between opposing political ideologies. Using NLP techniques, we identify key discussion topics and analyze sentiment patterns to gain insights into political discourse on social media.

## Features

- Topic modeling using Latent Dirichlet Allocation (LDA)
- Sentiment analysis of tweets
- Visualization of sentiment distribution across topics and political affiliations

## Methodology

1. Data collection from Twitter
2. Topic identification using LDA
3. Topic generalization with Large Language Models (LLM)
4. Sentiment analysis on processed tweets
5. Visualization of results

## Potential Applications

- Identifying common ground between opposing political groups
- Tracking sentiment shifts over time
- Guiding more effective political communication
- Predicting emerging trends in public opinion

## Future Enhancements

- Predictive political trend analysis
- Cross-platform opinion mining
- Real-time policy impact assessment

## Installation

This project uses [uv](https://github.com/astral-sh/uv) package manager. So, first please install uv using following [these instructions](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

To install the required packages, run the following command:

```bash
uv sync
```

Now, just run the following command to start the project

```bash
uv run streamlit run app.py
```

## Team

Project developed by Team NP: Suyog Dahal and Avaya Kumar Baniya
