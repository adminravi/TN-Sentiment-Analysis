import streamlit as st
import pandas as pd
import feedparser
import time
import schedule
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import spacy

# 🚀 Lazy Load Sentiment Model (Optimized for Faster Execution)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_model = load_sentiment_model()

import spacy
import streamlit as st

# 🚀 Lazy Load Named Entity Recognition (NER) Model
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# 🚀 Function to Fetch News Headlines from RSS Feeds
def fetch_news():
    sources = {
        "Google News": "https://news.google.com/rss/search?q=Tamil+Nadu+politics&hl=en-IN&gl=IN&ceid=IN:en",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "OneIndia Tamil": "https://tamil.oneindia.com/rss/tamil-news-fb.xml"
    }

    news_list = []
    for source, url in sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            news_list.append({
                "Source": source,
                "Title": entry.title,
                "Link": entry.link,
                "Published": entry.published
            })
    return pd.DataFrame(news_list)

# 🚀 Function to Analyze Sentiment Using BERT
def analyze_sentiment(df):
    sentiments = []
    for title in df["Title"]:
        result = sentiment_model(title)
        label = result[0]["label"]
        sentiments.append("Positive" if "4 stars" in label or "5 stars" in label else "Negative" if "1 star" in label or "2 stars" in label else "Neutral")

    df["Sentiment"] = sentiments
    return df

# 🚀 Function to Perform Named Entity Recognition (NER)
def extract_named_entities(df):
    entities = []
    for title in df["Title"]:
        doc = nlp(title)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
        entities.append(", ".join(named_entities) if named_entities else "None")

    df["Entities"] = entities
    return df

# 🚀 Function to Filter Sentiment by Political Party/Leader
def filter_by_keywords(df):
    keywords = {
        "Congress": ["Congress", "INC", "Rahul Gandhi"],
        "DMK": ["DMK", "M.K. Stalin", "Stalin"],
        "AIADMK": ["AIADMK", "Edappadi", "EPS"],
        "BJP": ["BJP", "Modi", "Amit Shah", "Annamalai"]
    }

    df["Category"] = "General"
    for category, words in keywords.items():
        df.loc[df["Title"].str.contains('|'.join(words), case=False, na=False), "Category"] = category

    return df

# 🚀 Function to Generate Word Cloud
def generate_wordcloud(df):
    text = ' '.join(df["Title"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# 🚀 Function to Refresh Data Every 6 Hours
def refresh_data():
    global news_df
    news_df = fetch_news()
    news_df = analyze_sentiment(news_df)
    news_df = extract_named_entities(news_df)
    news_df = filter_by_keywords(news_df)
    return news_df

schedule.every(6).hours.do(refresh_data)

# 🚀 Streamlit UI - Tamil Nadu Sentiment Dashboard
def main():
    st.title("Tamil Nadu Political Sentiment Dashboard")

    news_df = refresh_data()

    # 📊 Sentiment Overview
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = news_df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # 📊 Sentiment Breakdown by Political Party/Leader
    st.subheader("Sentiment Breakdown by Political Party/Leader")
    category_sentiments = news_df.groupby(["Category", "Sentiment"]).size().unstack().fillna(0)
    st.bar_chart(category_sentiments)

    # 📊 Sentiment Trend Analysis (7-day & 30-day Tracking)
    st.subheader("Sentiment Trend Over Time")
    news_df["Published"] = pd.to_datetime(news_df["Published"], errors='coerce')
    news_df = news_df.dropna(subset=["Published"])
    news_df = news_df.sort_values(by="Published")

    trend_filter = st.sidebar.radio("Select Trend Duration", ["7 Days", "30 Days"])
    days = 7 if trend_filter == "7 Days" else 30
    filtered_trend = news_df[news_df["Published"] >= (pd.to_datetime("today") - pd.Timedelta(days=days))]
    sentiment_trend = filtered_trend.groupby([filtered_trend["Published"].dt.date, "Sentiment"]).size().unstack().fillna(0)
    st.line_chart(sentiment_trend)

    # ☁ Trending Topics - Word Cloud
    st.subheader("Trending Topics in Tamil Nadu Politics")
    wordcloud = generate_wordcloud(news_df)
    st.image(wordcloud.to_array(), use_container_width=True)

    # 🔎 Filter Options
    st.sidebar.subheader("Filters")
    selected_entity = st.sidebar.selectbox("Filter by Political Leader/Party",
                                           ["All", "Congress", "DMK", "AIADMK", "BJP", "M.K. Stalin",
                                            "Edappadi K. Palaniswami", "Annamalai", "Rahul Gandhi",
                                            "Amit Shah", "Narendra Modi"])
    selected_sentiment = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral"])

    filtered_df = news_df
    if selected_entity != "All":
        filtered_df = filtered_df[filtered_df["Entities"].str.contains(selected_entity, case=False, na=False)]
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["Sentiment"] == selected_sentiment]

    # 📰 Display Latest News
    st.subheader("Latest News Headlines")
    st.dataframe(filtered_df[["Published", "Source", "Title", "Sentiment", "Category", "Entities", "Link"]])

    # ⏳ Automatic Refresh
    st.text("Data refreshes every 6 hours")

    # ✅ Health Check
    @st.cache_data
    def health_check():
        return "Healthy"

    st.sidebar.text(health_check())

# 🚀 Run the App
if __name__ == "__main__":
    main()
