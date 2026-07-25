# =========================================================
# Multi-Source Sentiment Analysis for Stocks & Crypto
# Professional Version
# =========================================================

# IMPORTS
import os
import re
import html
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import torch
import praw
import pandas as pd
import yfinance as yf
import snscrape.modules.twitter as sntwitter
from newsapi import NewsApiClient
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from functools import wraps
from pathlib import Path
import hashlib

# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

START_TIME = time.time()
# API CONFIGURATION
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT = "sentiment_app"
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"


# APPLICATION CONFIGURATION
@dataclass
class Config:
    # Asset
    asset_name: str = "Bitcoin"
    symbol: str = "BTC"
    market_type: str = "crypto"

    # Data
    days: int = 7
    limit: int = 300

    # Model
    model_name: str = "finbert"
    batch_size: int = 32
    confidence_threshold: float = 0.80
    
    # Available Models
    models: Dict[str, str] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = {"roberta":"cardiffnlp/twitter-roberta-base-sentiment",
                "finbert":"ProsusAI/finbert",
                "distilbert":"distilbert-base-uncased-finetuned-sst-2-english",
                "deberta":"microsoft/deberta-v3-base"}

CONFIG = Config()

# DEVICE CONFIGURATION
DEVICE = 0 if torch.cuda.is_available() else -1
logging.info(f"Running device: {'GPU' if DEVICE == 0 else 'CPU'}")

# TEXT PREPROCESSING ENGINE
def clean_text(text: str) -> str:
    """
    Clean raw text before sentiment analysis.
    Operations:
    - Decode HTML
    - Remove URLs
    - Remove mentions
    - Normalize hashtags
    - Remove extra spaces
    """

    if not text:
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove user mentions
    text = re.sub(r"@\w+", "", text)
    # Keep hashtag word but remove #
    text = re.sub(r"#", "", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# REMOVE EMPTY TEXTS
def remove_empty_texts(texts: List[str]) -> List[str]:
    """
    Remove empty documents.
    """
    return [text for text in texts if text and text.strip()]
    
# REMOVE DUPLICATES
def remove_duplicate_texts(texts: List[str]) -> List[str]:
    """
    Remove duplicate documents.
    """
    return list(dict.fromkeys(texts))
    
# TEXT LENGTH FILTER
def filter_short_texts(texts: List[str], min_length: int = 10) -> List[str]:
    """
    Remove very short texts
    that usually contain no sentiment.
    """
    return [text for text in texts if len(text) >= min_length]
    
# COMPLETE PREPROCESS PIPELINE
def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Complete NLP preprocessing pipeline.
    """
    logging.info(f"Raw documents: {len(texts)}")
    texts = [clean_text(text) for text in texts]
    texts = remove_empty_texts(texts)
    texts = remove_duplicate_texts(texts)
    texts = filter_short_texts(texts)
    
    logging.info(f"Clean documents: {len(texts)}")
    return texts


# DATA COLLECTION ENGINE
# TWITTER / X FETCHER
def retry(retries: int = 3, delay: int = 2):
    """
    Retry failed function calls.
    """
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return func( *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(f"{func.__name__} failed " f"attempt {attempt + 1}/{retries}")
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def fetch_twitter(query: str, days: int, limit: int) -> List[str]:
    """
    Collect posts from X(Twitter)
    using snscrape.
    """
    texts = []
    
    try:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        scraper = sntwitter.TwitterSearchScraper(f"{query} lang:en since:{since} -filter:retweets")
        for index, tweet in enumerate(scraper.get_items()):
            if index >= limit:
                break
            texts.append(tweet.content)
        logging.info(f"Twitter collected: {len(texts)}")
    except Exception as e:
        logging.error(f"Twitter fetch failed: {e}")
    return texts

# REDDIT FETCHER
def retry(retries: int = 3, delay: int = 2):
    """
    Retry failed function calls.
    """
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return func( *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(f"{func.__name__} failed " f"attempt {attempt + 1}/{retries}")
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def fetch_reddit(query: str, limit: int) -> List[str]:
    """Collect Reddit posts."""
    texts = []
    
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT)
        posts = reddit.subreddit("all").search( query, limit=limit)
        for post in posts:
            content = ( post.title + " " + post.selftext)
            texts.append(content)
        logging.info(
            f"Reddit collected: {len(texts)}")
    except Exception as e:
        logging.error(f"Reddit fetch failed: {e}")
    return texts

# NEWS API FETCHER
def retry(retries: int = 3, delay: int = 2):
    """
    Retry failed function calls.
    """
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return func( *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(f"{func.__name__} failed " f"attempt {attempt + 1}/{retries}")
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def fetch_news(query: str, days: int) -> List[str]:
    """Collect financial news
    from NewsAPI."""
    texts = []
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        response = newsapi.get_everything( q=query, from_param=from_date, language="en", sort_by="relevancy")
        for article in response["articles"]:
            title = article.get("title", "")
            description = article.get("description", "")
            texts.append(title + " " + description)
        logging.info(f"News collected: {len(texts)}")
    except Exception as e:
        logging.error(f"News fetch failed: {e}")
    return texts


# MODEL MANAGEMENT ENGINE
# MODEL LOADER
def load_sentiment_model():
    """
    Load transformer sentiment model.
    """
    model_path = CONFIG.models[CONFIG.model_name]
    logging.info(f"Loading NLP model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    sentiment_model = pipeline(task="sentiment-analysis",model=model,
            tokenizer=tokenizer, device=DEVICE, truncation=True, max_length=512)
    logging.info("Model loaded successfully")
    return sentiment_model

# INITIALIZE MODEL
sentiment_pipeline = load_sentiment_model()
# MODEL WARMUP
def warmup_model():
    """
    Run initial inference to initialize model.
    """
    try:
        sentiment_pipeline(["Market initialization"],
            batch_size=1)
        logging.info("Model warmup completed")
    except Exception as e:
        logging.warning(f"Warmup failed: {e}")
warmup_model()

def batch_predict(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        output = sentiment_pipeline(batch, batch_size=batch_size)
        results.extend(output)
    return results

def benchmark_model(texts):
    start_time = time.time()
    results = batch_predict(texts,
        CONFIG.batch_size)
    elapsed = time.time() - start_time
    return {"documents": len(texts),
        "execution_time": round(elapsed, 3),
        "documents_per_second": round(len(texts) / elapsed, 2)}

# SENTIMENT ANALYSIS ENGINE
# LABEL NORMALIZATION
def normalize_label(label: str) -> str:
    """
    Convert different model labels
    into common sentiment labels.
    """
    label = label.lower()
    mapping = {
        # RoBERTa
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        
        # FinBERT
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive"}
    return mapping.get(label, "neutral")

# SENTIMENT ANALYZER
def analyze_sentiment(texts: List[str]) -> Tuple[
    Dict[str, int], int, float, List[Dict[str, Any]]]:
    """
    Run sentiment analysis
    using Transformer model.
    """
    counts = {
        "positive": 0,
        "neutral": 0,
        "negative": 0}
    confidence_scores = []
    results = []
    if not texts:
        logging.warning("No texts available for analysis")
        return (counts, 0, 0.0, results)
    try:
        results = batch_predict(texts, CONFIG.batch_size)
        logging.info(f"Analyzed documents: {len(results)}")
        for result in results:
            label = normalize_label(result["label"])
            confidence = result["score"]
            counts[label] += 1
            confidence_scores.append(confidence)
    except Exception as e:
        logging.error(f"Sentiment inference failed: {e}")
        return (counts, 0, 0.0,[])
    
    # Positive - Negative balance
    sentiment_score = ( counts["positive"] - counts["negative"])
    
    # Average model confidence
    average_confidence = (sum(confidence_scores) / len(confidence_scores)
        if confidence_scores
        else 0.0)
    return (counts, sentiment_score, average_confidence, results)


# MODEL EVALUATION ENGINE
def evaluate_model(
    true_labels: List[str], predictions: List[str]) -> Dict[str, Any]:
    """
    Evaluate sentiment model performance.
    """
    
    if not true_labels or not predictions:
        logging.warning("No evaluation data available")
        return {}
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(true_labels, predictions)
    metrics = {"accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()}
    return metrics

def extract_predictions(results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract normalized labels
    from transformer outputs.
    """
    predictions = []
    for result in results:
        label = normalize_label(result["label"])
        predictions.append(label)
    return predictions

test_labels = [
    "positive",
    "negative",
    "neutral",
    "positive"]
test_texts = ["Bitcoin price is rising strongly",
    "Market crash expected soon",
    "Bitcoin remains stable",
    "Crypto adoption is increasing"]

# MARKET DATA ENGINE
def retry(retries: int = 3, delay: int = 2):
    """
    Retry failed function calls.
    """
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return func( *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(f"{func.__name__} failed " f"attempt {attempt + 1}/{retries}")
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def get_market_data(symbol: str, market_type: str, days: int) -> pd.DataFrame:
    """
    Fetch historical market data.
    """
    try:
        if market_type == "stock":
            ticker = symbol
        else:
            ticker = f"{symbol}-USD"
        data = yf.Ticker(ticker).history(period=f"{days}d")
        if data.empty:
            raise ValueError("No market data found")
        logging.info(f"Market data loaded: {len(data)} rows")
        return data
    except Exception as e:
        logging.error(f"Market data failed: {e}")
        return pd.DataFrame()
    
# MARKET FEATURE ENGINE
def calculate_market_features(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate market statistics.
    """
    if data.empty:
        return {}
    close = data["Close"]
    current_price = float(close.iloc[-1])
    first_price = float(close.iloc[0])
    change_percent = ((current_price - first_price) / first_price * 100)
    volatility = float(close.pct_change().std() * 100)
    features = {"current_price": round(current_price, 4),
        "change_percent": round(change_percent, 2),
        "volatility": round(volatility, 2)}
    return features

# SIGNAL ENGINE
def generate_market_signal(sentiment_score: int, confidence: float, market_features: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate combined market signal.
    """
    score = 0
    # Sentiment component
    if sentiment_score > 20:
        score += 1
    elif sentiment_score < -20:
        score -= 1
    # Price momentum component
    change = market_features.get("change_percent", 0)
    if change > 2:
        score += 1
    elif change < -2:
        score -= 1
    # Confidence filter
    if confidence < CONFIG.confidence_threshold:
        signal = "LOW_CONFIDENCE"
    elif score >= 2:
        signal = "BULLISH"
    elif score <= -2:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"
    return {"signal": signal,
        "combined_score": score,
        "model_confidence": round(confidence, 4)}
    
# REPORT GENERATOR
def generate_report(
    asset_name: str,
    symbol: str,
    model_name: str,
    counts: Dict[str, int],
    sentiment_score: int,
    confidence: float,
    market_features: Dict[str, float],
    signal: Dict[str, Any], 
    execution_time: float) -> Dict[str, Any]:
    """
    Generate final analysis report.
    """
    total_documents = sum(counts.values())
    percentages = calculate_percentages(counts)
    report = {"asset": asset_name,
        "symbol": symbol,
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "documents": {"total": total_documents,
        "positive": counts["positive"],
        "neutral": counts["neutral"],
        "negative": counts["negative"]},
        "sentiment": {"score": sentiment_score,
        "confidence": round(confidence, 4),
        "distribution": percentages},
        "market": market_features,
        "signal": signal,
        "system": {"execution_time_seconds": round(execution_time, 3)}}
    return report

# SENTIMENT PERCENTAGE
def calculate_percentages(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate sentiment distribution.
    """
    total = sum(counts.values())
    if total == 0:
        return {
            "positive": 0,
            "neutral": 0,
            "negative": 0}
    return {key: round(value / total * 100, 2)
        for key, value in counts.items()}
    
# JSON EXPORT
def save_json(data: Dict[str, Any], filename: str = "sentiment_report.json"):
    """
    Save report as JSON.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    logging.info(f"JSON report saved: {filename}")
    
# CSV EXPORT
def save_csv(data: Dict[str, Any], filename: str = "sentiment_report.csv"):
    """
    Save flattened report to CSV.
    """
    df = pd.json_normalize(data)
    df.to_csv(filename, index=False)
    logging.info(f"CSV report saved: {filename}")
    
    
# MAIN PIPELINE
def main():
    try:
        pipeline_start = time.time()
        logging.info("Starting Financial Sentiment Pipeline")
        
        # 1. DATA COLLECTION
        query = (f"{CONFIG.asset_name} "
            f"OR "
            f"{CONFIG.symbol}")
        twitter_data = fetch_twitter(query,
            CONFIG.days,
            CONFIG.limit)
        reddit_data = fetch_reddit( query, CONFIG.limit // 3)
        news_data = fetch_news( query, CONFIG.days)
        all_texts = (twitter_data + reddit_data + news_data)
        if not all_texts:
            raise RuntimeError("No documents collected")
            
        # 2. TEXT PREPROCESSING
        clean_documents = preprocess_texts(all_texts)
        benchmark = benchmark_model(clean_documents)
        logging.info(benchmark)
        if not clean_documents:
            raise RuntimeError("No valid documents after cleaning")
            
        # 3. SENTIMENT ANALYSIS
        (sentiment_counts, sentiment_score, confidence, sentiment_results)= analyze_sentiment(clean_documents) = analyze_sentiment(clean_documents)
        
        # 4. MARKET DATA
        market_data = get_market_data(
            CONFIG.symbol,
            CONFIG.market_type,
            CONFIG.days)
        market_features = calculate_market_features(market_data)
        
        # 5. SIGNAL GENERATION
        signal = generate_market_signal(sentiment_score, confidence, market_features)
        
        # 6. FINAL REPORT
        execution_time = (time.time() - pipeline_start)
        report = generate_report(
            asset_name=CONFIG.asset_name,
            symbol=CONFIG.symbol,
            model_name=CONFIG.model_name,
            counts=sentiment_counts,
            sentiment_score=sentiment_score,
            confidence=confidence,
            market_features=market_features,
            signal=signal,
            execution_time=execution_time)
        
        # 7. EXPORT
        del sentiment_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_json(report)
        save_csv(report)
        logging.info("Pipeline completed successfully")
        print(json.dumps(report, indent=4, ensure_ascii=False))
    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        
# APPLICATION ENTRY POINT
if __name__ == "__main__":
    main()
    
# RETRY SYSTEM
def retry(retries: int = 3, delay: int = 2):
    """
    Retry failed function calls.
    """
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return func( *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(f"{func.__name__} failed " f"attempt {attempt + 1}/{retries}")
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# CACHE SYSTEM
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def generate_cache_key(name: str, value: str):
    key = (name + value)
    return hashlib.md5(key.encode()).hexdigest()
def save_cache(key: str, data: Any):
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        
def load_cache(key: str):
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)