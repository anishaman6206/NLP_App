# NLP App  

Welcome to the **NLP App**!  

## 🚀 Features  

### 1. **Language Detection**  
Detect the language of any text input with high accuracy.  
- Ideal for multilingual datasets and cross-lingual applications.

### 2. **Emotion Prediction**  
Predict the emotional tone of a given text.  
- Categories include happiness, sadness, anger, fear, and more.  

### 3. **Sentiment Analysis**  
Analyze the sentiment (positive, negative, or neutral) and tone of textual content.  

### 4. **Real-Time Translation**  
Translate text from any source language to a target language instantly.  

### 5. **Text Summarization**  
Generate concise summaries of large textual content.  

---

## 🛠️ Technologies Used  

1. **Backend**: [Flask](https://flask.palletsprojects.com/) - A lightweight WSGI web application framework for Python.  
2. **NLP Models**: Pre-trained models from [Hugging Face Transformers](https://huggingface.co/) like:  
   - `bert-base-uncased-emotion`
   - `finiteautomata/bertweet-base-sentiment-analysis`
   - `Helsinki-NLP/opus-mt-en-hi`
   - `xlm-roberta-base-language-detection`  
3. **APIs**: Hugging Face Inference API for seamless model integration.  
4. **Frontend**: HTML, CSS for a responsive UI.  

---

## 🌟 How to Use  

### Locally

#### 1. Clone the Repository  
```bash  
git clone https://github.com/anishaman6206/NLP_App.git  
cd NLP_App  
```  

#### 2. Set Up Virtual Environment
```bash  
python3 -m venv venv  
source venv/bin/activate  # For Windows: venv\Scripts\activate  
```  

#### 3. Install Dependencies  
```bash  
pip install -r requirements.txt  
```  

#### 4. Run the App  
```bash  
python app.py  
```  
Navigate to `http://127.0.0.1:5000` in your browser.

---


## 📂 File Structure  

```plaintext  
nlp-app/  
├── static/  
│   └── styles.css       # Styles for the frontend  
├── templates/  
│   ├── login.html       # Login page  
│   ├── register.html    # Registration page  
│   ├── profile.html     # User dashboard  
│   ├── sentiment_analysis.html  
│   ├── emotion_prediction.html  
│   ├── language_detection.html  
│   └── language_translation.html  
├── app.py               # Flask app entry point  
├── api.py               # API integration logic  
├── db.py                # Database integration logic  
├── requirements.txt     # List of dependencies 
├── users.json   
└── README.md            # Project documentation  
```  

---

