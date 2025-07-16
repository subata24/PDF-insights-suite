# 📄 PDF Insights Suite

Turn any PDF into intelligent, structured insights — powered by NLP.

> 👩‍💻 Built with 💖 by **Subata**  
> 📊 Ideal for Resumes, Research Papers, Business Reports, and more.

---

## 🚀 Live Demo

🔗 **[Try it on Streamlit ](https://pdf-insights-suite-v6g8fizn2n9upxyrs2c5gf.streamlit.app/)**  


---

## 🧠 Features

- 📃 **PDF Text Extraction**  
- 📊 **Word Stats & Graphs**  
- 🔍 **Keyword Extraction (RAKE)**  
- 📝 **Text Summarization (LSA)**  
- 🏷️ **Named Entity Recognition (spaCy)**  
- 💬 **Sentiment Analysis (VADER)**  
- 📥 **Export as CSV / JSON**  
- 🧭 **Use-Case Modes** (Resume, Report, Research, General)

---

## 🎯 Who Is This For?

- ✅ **Freelancers** — offer resume analysis, summarization gigs  
- ✅ **Students** — extract key points from papers  
- ✅ **HR Teams** — screen resumes by keywords and tone  
- ✅ **Writers** — analyze sentiment and structure  
- ✅ **Anyone** — needing quick insights from PDFs

---

## 💻 Built With

- **Streamlit** – frontend UI  
- **PyMuPDF (fitz)** – PDF text extraction  
- **RAKE-NLTK** – keyword extraction  
- **Sumy** – document summarization  
- **spaCy** – named entity recognition  
- **VADER** – sentiment analysis  
- **pandas**, **matplotlib** – data handling & visualization

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/pdf-insights-suite.git
cd pdf-insights-suite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLP data
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm

# 4. Run the app
streamlit run app.py
