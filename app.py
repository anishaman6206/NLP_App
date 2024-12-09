from flask import Flask, render_template, request, redirect, session
from db import Database
from api import API

app = Flask(__name__, static_url_path='/static')


# Set a secret key for session management
app.secret_key = 'nlpapp_12345'

dbo = Database()
apio = API()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/perform_registration', methods=['post'])
def perform_registration():
    name = request.form.get('user_name')
    email = request.form.get('user_email')
    password = request.form.get('user_password')

    response = dbo.insert(name, email, password)

    if response:
        return render_template('login.html', message='Registration Successful. Kindly login to proceed')
    else:
        return render_template('register.html', message='Email already exists')

@app.route('/perform_login', methods=['post'])
def perform_login():
    email = request.form.get('user_email')
    password = request.form.get('user_password')

    response = dbo.search(email, password)

    if response:
        session['logged_in'] = 1  # Set session for logged-in user
        return redirect('/profile')  # Redirect to profile after successful login
    else:
        return render_template('login.html', message='Incorrect email or password')

@app.route('/profile')
def profile():
    # Check if the user is logged in, else redirect to login page
    if session.get('logged_in') != 1:
        return redirect('/')  # Redirect to login page if not logged in
    return render_template('profile.html')

@app.route('/sentiment_analysis')
def sentiment_analysis():
    # Check if the user is logged in, else redirect to login page
    if session.get('logged_in') != 1:
        return redirect('/')
    return render_template('sentiment_analysis.html')

@app.route('/analyze_sentiment', methods=['post'])
def analyze_sentiment():
    text = request.form.get('text')
    result = apio.sentiment_analysis(text)
    while "error" in result and "loading" in result["error"].lower():
        import time
        print(result["error"])
        print("Retrying in 3 seconds...")
        time.sleep(3)  # Wait for 5 seconds before retrying

    return render_template('sentiment_analysis.html', result=result)

@app.route('/emotion_prediction')
def emotion_prediction():
    # Check if the user is logged in, else redirect to login page
    if session.get('logged_in') != 1:
        return redirect('/')
    return render_template('emotion_prediction.html')

@app.route('/predict_emotion', methods=['post'])
def predict_emotion():
    text = request.form.get('text')
    result = apio.emotion_prediction(text)

    return render_template('emotion_prediction.html', result=result)

'''@app.route('/language_translation')
def language_translation():
    # Check if the user is logged in, else redirect to login page
    if session.get('logged_in') != 1:
        return redirect('/')
    return render_template('language_translation.html')

@app.route('/translate_en_hi', methods=['post'])
def translate_en_hi():
    text = request.form.get('text')
    result = apio.translate_en_to_hi(text)

    return render_template('language_translation.html', result=result)
'''
@app.route('/language_detection')
def language_detection():
    # Check if the user is logged in, else redirect to login page
    if session.get('logged_in') != 1:
        return redirect('/')
    return render_template('language_detection.html')

@app.route('/detect_lang', methods=['post'])
def detect_lang():
    text = request.form.get('text')
    result = apio.language_detection(text)

    return render_template('language_detection.html', result=result)

'''@app.route('/paraphrasing')
def paraphrasing():
    return render_template('paraphrasing.html')

@app.route('/perform_paraphrase', methods=['post'])
def perform_paraphrase():
    text = request.form.get('text')
    result = apio.paraphrase_text(text)
    return render_template('paraphrasing.html', result=result)

'''
@app.route('/summarization')
def summarization():
    return render_template('summarization.html')

@app.route('/perform_summarization', methods=['post'])
def perform_summarization():
    text = request.form.get('text')
    result = apio.summarize_text(text)
    return render_template('summarization.html', result=result)

'''
@app.route('/keyword_extraction')
def keyword_extraction():
    return render_template('keyword_extraction.html')

@app.route('/perform_keyword_extraction', methods=['post'])
def perform_keyword_extraction():
    text = request.form.get('text')
    result = apio.extract_keywords(text)
    return render_template('keyword_extraction.html', result=result)
'''
@app.route('/real_time_translation')
def real_time_translation():
    return render_template('real_time_translation.html')

@app.route('/perform_translation', methods=['post'])
def perform_translation():
    text = request.form.get('text')
    source_language = request.form.get('source_language')
    target_language = request.form.get('target_language')

    # Call the API for real-time translation
    result = apio.real_time_translation(text, source_language, target_language)

    return render_template('real_time_translation.html', result=result)


@app.route('/logout')
def logout():
    # Log out the user by clearing the session
    session.clear()
    return redirect('/')  # Redirect to login page after logging out

if __name__ == "__main__":
    app.run(debug=False)
