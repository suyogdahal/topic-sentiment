import streamlit as st
import pandas as pd
import plotly.express as px
from gensim.models.ldamodel import LdaModel

# got thsese topics from llm by feeding the result of lda:
# [[('great', 0.013737874), ('amp', 0.01289574), ('thanks', 0.008955392), ('today', 0.007472114), ('time', 0.0062110033), ('us', 0.0061538145), ('small', 0.005616049), ('forward', 0.00560348), ('students', 0.005460738), ('thank', 0.0054471525)], [('amp', 0.01625349), ('women', 0.01040878), ('americans', 0.008185313), ('families', 0.00798738), ('country', 0.007956142), ('people', 0.007874884), ('every', 0.007743965), ('american', 0.0074607567), ('must', 0.007100312), ('day', 0.006909426)], [('today', 0.016090466), ('thank', 0.011228169), ('honor', 0.008339805), ('great', 0.008195283), ('us', 0.008175254), ('hearing', 0.0074929986), ('th', 0.0070543974), ('day', 0.006695193), ('live', 0.006394901), ('join', 0.0061513176)], [('tax', 0.019043108), ('cuts', 0.006102945), ('jobs', 0.006076858), ('amp', 0.006058148), ('trump', 0.005697971), ('new', 0.0055173286), ('pay', 0.005269086), ('happy', 0.0051747956), ('states', 0.004544045), ('president', 0.00429671)], [('house', 0.015671087), ('bill', 0.011572243), ('act', 0.010448738), ('amp', 0.009365725), ('today', 0.008708077), ('congress', 0.007556709), ('proud', 0.0061408076), ('vote', 0.005947531), ('legislation', 0.0058746794), ('health', 0.0058271973)]]
lda_topic_names = {
    0: "Appreciation and Gratitude",
    1: "American Families and Values",
    2: "Civic Engagement and Honor",
    3: "Tax Policy and Economic Issues",
    4: "Legislative Action and Health Care",
}


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


# Load data
data = pd.read_csv("data/processed.csv")

# Load LDA model
lda_model = load_lda_model("lda_model/lda_model.gensim")

# Streamlit app title
st.title("Tweet Topic and Sentiment Analysis")

# Topic Visualization
st.subheader("Topic Visualization")
topics = lda_model.show_topics(formatted=False)
topic_list = [topic[1] for topic in topics]
topic_list = list(lda_topic_names.values())

# Select a topic to analyze
selected_topic = st.selectbox("Select a Topic to Analyze", options=topic_list)

# Filter data based on selected topic
filtered_data = data[data["Assigned Topic"] == selected_topic]

# Sentiment over time for the selected topic
st.subheader(f"Sentiment Over Time for Topic: {selected_topic}")
for party in filtered_data["Party"].unique():
    party_data = filtered_data[filtered_data["Party"] == party]
    fig = px.line(
        party_data,
        x="timestamp",
        y="Sentiment",
        title=f"Sentiment Over Time for {party}",
    )
    st.plotly_chart(fig)

# Additional components (if needed)
st.subheader("Overall Sentiment Distribution")
fig = px.histogram(
    data, x="Sentiment", nbins=50, title="Overall Sentiment Distribution"
)
st.plotly_chart(fig)

st.subheader("Topic Distribution")
fig = px.bar(data, x="Assigned Topic", title="Topic Distribution")
st.plotly_chart(fig)
