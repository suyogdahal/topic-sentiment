import streamlit as st
import pandas as pd
import plotly.express as px
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def load_lda_model(model_path):
    """
    Load a pre-trained LDA model from the specified path.

    Parameters:
    - model_path (str): Path to the LDA model file.

    Returns:
    - LdaModel: The loaded LDA model.
    """
    lda_model = LdaModel.load(model_path)
    return lda_model


COLOR_MAP = {"Republican": "red", "Democrat": "blue"}

# Load data
data = pd.read_csv("data/processed.csv")
data["Year"] = pd.to_datetime(data["timestamp"]).dt.year


# Load LDA model
lda_model = load_lda_model("lda_model/lda_model.gensim")

# Streamlit app title
st.title("Tweet Topic and Sentiment Analysis")

# Topic Visualization
st.subheader("General Information")
topics = lda_model.show_topics(formatted=False)
topic_list = [topic[1] for topic in topics]


# got thsese topics from llm by feeding the result of lda:
# [[('great', 0.013737874), ('amp', 0.01289574), ('thanks', 0.008955392), ('today', 0.007472114), ('time', 0.0062110033), ('us', 0.0061538145), ('small', 0.005616049), ('forward', 0.00560348), ('students', 0.005460738), ('thank', 0.0054471525)], [('amp', 0.01625349), ('women', 0.01040878), ('americans', 0.008185313), ('families', 0.00798738), ('country', 0.007956142), ('people', 0.007874884), ('every', 0.007743965), ('american', 0.0074607567), ('must', 0.007100312), ('day', 0.006909426)], [('today', 0.016090466), ('thank', 0.011228169), ('honor', 0.008339805), ('great', 0.008195283), ('us', 0.008175254), ('hearing', 0.0074929986), ('th', 0.0070543974), ('day', 0.006695193), ('live', 0.006394901), ('join', 0.0061513176)], [('tax', 0.019043108), ('cuts', 0.006102945), ('jobs', 0.006076858), ('amp', 0.006058148), ('trump', 0.005697971), ('new', 0.0055173286), ('pay', 0.005269086), ('happy', 0.0051747956), ('states', 0.004544045), ('president', 0.00429671)], [('house', 0.015671087), ('bill', 0.011572243), ('act', 0.010448738), ('amp', 0.009365725), ('today', 0.008708077), ('congress', 0.007556709), ('proud', 0.0061408076), ('vote', 0.005947531), ('legislation', 0.0058746794), ('health', 0.0058271973)]]
lda_topic_names = {
    1: "Appreciation and Gratitude",
    2: "American Families and Values",
    3: "Civic Engagement and Honor",
    4: "Tax Policy and Economic Issues",
    5: "Legislative Action and Health Care",
}


data["Assigned Topic Map"] = data["Assigned Topic"].map(lda_topic_names)
# Select a topic to analyze


fig = px.histogram(
    data,
    x="Party",
    color="Party",
    nbins=50,
    title="Overall Tweets Distribution",
    color_discrete_map=COLOR_MAP,
)
st.plotly_chart(fig)

# st.subheader("Topic Distribution")
fig = px.bar(data, x="Assigned Topic Map", title="Overall Topic Distribution")
st.plotly_chart(fig)

# st.subheader("Overall Sentiment Distribution")
fig = px.histogram(
    data, x="Sentiment", nbins=50, title="Overall Sentiment Distribution"
)
st.plotly_chart(fig)

selected_topic = st.selectbox(
    "Select a Topic to Analyze", options=list(lda_topic_names.values())
)

# Generate word cloud for the selected topic
st.subheader(f"Word Cloud for Topic: {selected_topic}")
selected_topic_index = list(lda_topic_names.values()).index(selected_topic)
selected_topic_words = dict(lda_model.show_topic(selected_topic_index, topn=100))
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(selected_topic_words)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Filter data based on selected topic
filtered_data = data[data["Assigned Topic Map"] == selected_topic]

# Calculate the ratio of positive to negative sentiments for each party
ratios = []
for party in filtered_data["Party"].unique():
    party_data = filtered_data[filtered_data["Party"] == party]
    sentiment_counts = (
        party_data.groupby(["Year", "Sentiment"]).size().unstack(fill_value=0)
    )
    sentiment_counts["Ratio"] = (
        sentiment_counts["POSITIVE"] / sentiment_counts["NEGATIVE"]
    )
    sentiment_counts["Party"] = party
    ratios.append(sentiment_counts.reset_index())

# Combine the data for all parties
combined_ratios = pd.concat(ratios)

# Plot the ratio over time for different parties in the same plot
fig = px.line(
    combined_ratios,
    x="Year",
    y="Ratio",
    color="Party",
    line_dash="Party",
    line_shape="linear",
    color_discrete_map=COLOR_MAP,
    title="Ratio of Positive to Negative Sentiments Over Time by Party",
)
st.plotly_chart(fig)

# Sentiment grouped by years and parties
st.subheader(f"Sentiment Grouped by Years for Topic: {selected_topic}")


fig = px.bar(
    filtered_data,
    x="Year",
    y="Sentiment",
    color="Party",
    barmode="group",
    title=f"Sentiment Grouped by Years for Topic: {selected_topic}",
    color_discrete_map=COLOR_MAP,
)
st.plotly_chart(fig)

# Calculate the percentage of total tweets for each year and party
tweet_counts = (
    filtered_data.groupby(["Year", "Party"]).size().reset_index(name="Tweet Count")
)
total_tweets_per_year = (
    filtered_data.groupby("Year").size().reset_index(name="Total Tweets")
)
tweet_counts = tweet_counts.merge(total_tweets_per_year, on="Year")
tweet_counts["Percentage"] = (
    tweet_counts["Tweet Count"] / tweet_counts["Total Tweets"]
) * 100

# Plot the bar chart showing the comparison of the percentage of total tweets
fig = px.bar(
    tweet_counts,
    x="Year",
    y="Percentage",
    color="Party",
    barmode="group",
    title=f"Percentage of Total Tweets Grouped by Years for Topic: {selected_topic}",
    color_discrete_map=COLOR_MAP,
)
st.plotly_chart(fig)

# Calculate the percentage of positive and negative tweets for each year and party
sentiment_counts = (
    filtered_data.groupby(["Year", "Party", "Sentiment"])
    .size()
    .reset_index(name="Tweet Count")
)
total_tweets_per_year_party = (
    filtered_data.groupby(["Year", "Party"]).size().reset_index(name="Total Tweets")
)
sentiment_counts = sentiment_counts.merge(
    total_tweets_per_year_party, on=["Year", "Party"]
)
sentiment_counts["Percentage"] = (
    sentiment_counts["Tweet Count"] / sentiment_counts["Total Tweets"]
) * 100

# Create a custom color mapping for party and sentiment
custom_color_map = {
    ("Republican", "POSITIVE"): "blue",
    ("Republican", "NEGATIVE"): "darkblue",
    ("Democrat", "POSITIVE"): "red",
    ("Democrat", "NEGATIVE"): "darkred",
}

# Add a new column for combined party and sentiment
sentiment_counts["Party_Sentiment"] = sentiment_counts.apply(
    lambda row: (row["Party"], row["Sentiment"]), axis=1
)

# Plot the stacked bar chart showing the percentage of positive and negative tweets side by side for each party
fig = px.bar(
    sentiment_counts,
    x="Year",
    y="Percentage",
    color="Party_Sentiment",
    barmode="stack",
    facet_col="Party",
    title=f"Percentage of Positive and Negative Tweets Grouped by Years for Topic: {selected_topic}",
    color_discrete_map=custom_color_map,
)
st.plotly_chart(fig)
