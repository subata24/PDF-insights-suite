import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import json
from io import StringIO, BytesIO

# NLP tools
from rake_nltk import Rake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import spacy
nlp = spacy.blank("en")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Downloads
nltk.download('stopwords')
#nlp = spacy.load("en_core_web_sm")

# --- Streamlit Page Settings ---
st.set_page_config(page_title="PDF Insights Suite", layout="wide")

# --- Sidebar Branding ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4727/4727411.png", width=100)
    st.markdown("Turn any PDF into intelligent insight.")
    use_case = st.selectbox(
        "🧠 Choose Use Case",
        ["Resume Analysis", "Research Article", "Business Report", "General Document"]
    )

# --- App Header ---
st.title(f"📄 PDF Insights: {use_case}")

if use_case == "Resume Analysis":
    st.markdown("🧑‍💼 This mode focuses on skills, keywords, and tone in resumes.")
elif use_case == "Research Article":
    st.markdown("🔬 Focus on summaries, entities, and academic keywords.")
elif use_case == "Business Report":
    st.markdown("📊 Key highlights and tone detection for strategic docs.")
else:
    st.markdown("📚 Analyze any PDF for keywords, sentiment, and summary.")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

# === Helper Functions ===

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_word_stats(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    reading_time = round(word_count / 200, 2)
    return word_count, reading_time, words

def plot_word_freq(words, top_n=10):
    word_freq = Counter(words)
    common = word_freq.most_common(top_n)
    df = pd.DataFrame(common, columns=["Word", "Frequency"])
    st.bar_chart(df.set_index("Word"))

def extract_keywords(text, num_keywords=10):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:num_keywords]

def generate_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

#def extract_entities(text):
    #doc = nlp(text)
   # return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores['compound']
    if polarity > 0.1:
        sentiment = "Positive 😊"
    elif polarity < -0.1:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
    return sentiment, polarity, scores

def generate_download_files(keywords, summary, entities, sentiment_info):
    data = {
        "keywords": keywords,
        "summary": summary,
        "entities": [{"entity": e, "type": t} for e, t in entities],
        "sentiment": {
            "label": sentiment_info[0],
            "polarity": sentiment_info[1],
            "all_scores": sentiment_info[2]
        }
    }

    # JSON
    json_bytes = BytesIO(json.dumps(data, indent=2).encode("utf-8"))

    # CSV
    csv_buffer = StringIO()
    entity_df = pd.DataFrame(data["entities"])
    sentiment_df = pd.DataFrame([data["sentiment"]])
    full_df = pd.concat([entity_df, sentiment_df], axis=0, ignore_index=True)
    full_df.to_csv(csv_buffer, index=False)
    csv_bytes = BytesIO(csv_buffer.getvalue().encode("utf-8"))

    return json_bytes, csv_bytes

# === Main App Logic ===

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    word_count, reading_time, words = get_word_stats(text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📃 Extracted Text")
        st.text_area("Raw PDF Text", text, height=300)

    with col2:
        st.subheader("📊 Document Stats")
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Estimated Reading Time:** {reading_time} minutes")
        st.subheader("🔝 Top Words")
        plot_word_freq(words)

    # NLP Features
    st.markdown("---")
    st.header("🧠 NLP Insights")

    st.subheader("🔍 Top Keywords")
    keywords = extract_keywords(text)
    st.write(keywords)

    st.subheader("📝 Text Summary")
    summary = generate_summary(text)
    st.text_area("Summary", summary, height=200)

    st.subheader("🏷️ Named Entities")
    entities = extract_entities(text)
    if entities:
        st.dataframe(pd.DataFrame(entities, columns=["Entity", "Type"]))
    else:
        st.write("No named entities found.")

    st.subheader("💬 Sentiment Analysis")
    sentiment, polarity, scores = analyze_sentiment(text)
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Polarity Score:** {polarity}")
    st.write(f"**Details:** {scores}")

    # Export Options
    st.markdown("---")
    st.header("📥 Export Your Results")
    json_file, csv_file = generate_download_files(keywords, summary, entities, (sentiment, polarity, scores))

    st.download_button("📄 Download as JSON", data=json_file, file_name="pdf_insights.json", mime="application/json")
    st.download_button("📑 Download as CSV", data=csv_file, file_name="pdf_insights.csv", mime="text/csv")

    # Insights Summary
    st.markdown("---")
    st.header("📌 Key Takeaways")
    with st.expander("📋 Click to view summary of analysis"):
        st.markdown(f"- **Use Case:** {use_case}")
        st.markdown(f"- **Word Count:** {word_count}")
        st.markdown(f"- **Top Keywords:** {', '.join(keywords)}")
        st.markdown(f"- **Named Entities Found:** {len(entities)}")
        st.markdown(f"- **Sentiment:** {sentiment} (Score: {round(polarity, 2)})")
        st.markdown("- **Summary Preview:**")
        st.info(summary[:300] + "...")
