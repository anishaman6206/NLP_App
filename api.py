import time
from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer
# from gliner import GLiNER
import requests
import logging


class API:

    def __init__(self):
        # Log in using your Hugging Face token
        login(token="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")
        # Set the logging level to ERROR to suppress warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def sentiment_analysis(self, text):
        API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({"inputs": text[:250]})

        # Handle the case where the model is still loading
        while "error" in output and "loading" in output["error"].lower():
            print(output["error"])
            print("Retrying in 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            output = query({"inputs": text})

        # Print the output once the model is ready
        return output

    '''def ner(self,text,labels_list):

        model = GLiNER.from_pretrained("DeepMount00/GLiNER_PII_ITA")
        entities = model.predict_entities(text, labels_list)

        return entities'''

    def language_detection(self, text):
        model_name = "papluca/xlm-roberta-base-language-detection"

        # Load the pipeline for language detection
        language_detection_pipeline = pipeline("text-classification", model=model_name)

        detected_language = language_detection_pipeline(text)
        return detected_language

    def emotion_prediction(self, text):
        tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
        model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

        # Create pipeline
        emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        # Predict emotions
        emotions = emotion_classifier(text)
        return emotions

    def translate_en_to_hi(self, text):
        # Load the model and tokenizer
        model_name = "Helsinki-NLP/opus-mt-en-hi"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Example text

        # Tokenize and translate
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

        # Decode the translation
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return translated_text

    def paraphrase_text(self, text):
        model_name = "tuner007/pegasus_paraphrase"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize and generate paraphrase
        inputs = tokenizer("paraphrase: " + text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1, num_beams=5)

        # Decode the paraphrased text
        paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased_text

    def summarize_text(self, text):
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize and summarize
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
        summary = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)

        # Decode the summary
        summarized_text = tokenizer.decode(summary[0], skip_special_tokens=True)
        return summarized_text

    def extract_keywords(self, text):
        model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Load pipeline for keyword extraction
        keyword_extraction_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

        # Extract keywords
        keywords = keyword_extraction_pipeline(text)
        extracted_keywords = [keyword["word"] for keyword in keywords]
        return extracted_keywords


    def real_time_translation(self, text, source_language, target_language):
        model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"

    # Load the tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    # Tokenize and translate
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

    # Decode the translated text
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return translated_text






