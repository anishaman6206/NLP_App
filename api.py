from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer
import time
import requests
import logging

class API:
    def __init__(self):
        # Set logging level to suppress warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)
        self.token = "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"

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
            print("Retrying in 3 seconds...")
            time.sleep(3)  # Wait for 3 seconds before retrying
            output = query({"inputs": text})

        # Print the output once the model is ready
        return output

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








