# Multi-Source Sentiment Analysis for Stocks & Crypto

### Implemented a configurable Transformer-based NLP pipeline supporting multiple sentiment models including FinBERT and RoBERTa with GPU acceleration.

# 📊 Multi-Source Sentiment Analysis for Stocks & Cryptocurrencies

This project implements a **multi-source sentiment analysis engine** that analyzes public opinion from social media and news sources and combines it with **financial market data** for both **stocks** and **cryptocurrencies**.

The system is designed as a **research-grade pipeline** suitable for financial analysis, market sentiment tracking, and downstream applications such as trading signals or dashboards.

---

## 🚀 Features

- ✅ Multi-source text data collection:
  - Twitter (via `snscrape`, no API required)
  - Reddit (via official Reddit API)
  - News articles (via NewsAPI)
- ✅ Supports both:
  - **Stocks** (e.g. AAPL, TSLA)
  - **Cryptocurrencies** (e.g. BTC, ETH)
- ✅ Transformer-based sentiment analysis using **RoBERTa**
- ✅ Aggregated sentiment scoring (Positive − Negative)
- ✅ Human-readable **textual insight report**
- ✅ Clean, single-file Python implementation
- ✅ Ready for extension (API, dashboard, ML models)

---

## 🧠 Sentiment Model

- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Labels**:
  - `LABEL_0`: Negative
  - `LABEL_1`: Neutral
  - `LABEL_2`: Positive

**Sentiment Score Formula:**
---
# 📥 Data Sources

| Source   | Method     | API Required |
|--------|------------|--------------|
| Twitter | snscrape   | ❌ No |
| Reddit | PRAW       | ✅ Yes |
| News   | NewsAPI    | ✅ Yes |
| Prices | yfinance   | ❌ No |
---
# 🛠 Requirements
- Python **3.9+**
- Internet connection
- API keys for:
  - Reddit
  - NewsAPI

Install dependencies:

```bash
pip install -r requirements.txt
```

**Sample Output**
Asset: Bitcoin (BTC)
Market Price: 43120.45

Sentiment Summary:
- Positive: 182
- Neutral: 241
- Negative: 96

Overall Market Mood: Mildly Bullish

Interpretation:
Public discourse across social media and news sources suggests a mildly bullish
sentiment toward Bitcoin over the last 7 days.
