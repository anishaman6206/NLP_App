import time
from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
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

# Print the detected language



