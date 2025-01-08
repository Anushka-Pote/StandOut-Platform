# app.py

from flask import Flask, render_template, request, redirect, Response
from sqlalchemy.orm import declarative_base
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
from keras.preprocessing import image
nltk.data.path.append("/path/to/nltk_data")
nltk.download('popular')
import cv2
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('model.h5', compile=False) 
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
from flask import Flask, render_template, jsonify
from newsapi import NewsApiClient

from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.static_folder = 'static'

file_path = 'Ai&DS.csv'  
df = pd.read_csv(file_path)

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def display_tree(self, level=0):
        prefix = "  " * level
        return f"{prefix}- {self.value}" + "\n".join(child.display_tree(level + 1) for child in self.children)

difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}

def run_kmeans(subject, platform, difficulty, duration, rating):
    selected_subject_df = df[(df['Subject'] == subject) & (df['Platform'] == platform)]

    selected_subject_df['Difficulty'] = selected_subject_df['Difficulty'].map(difficulty_mapping)

    features = selected_subject_df[['Difficulty', 'Duration', 'Rating']]

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    kmeans = KMeans(n_clusters=4, random_state=50)
    selected_subject_df['Cluster'] = kmeans.fit_predict(features_scaled)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    selected_subject_df['PCA1'] = features_pca[:, 0]
    selected_subject_df['PCA2'] = features_pca[:, 1]

    user_input = pd.DataFrame({
        'Difficulty': [difficulty],
        'Duration': [duration],
        'Rating': [rating]
    })

    user_input['Difficulty'] = user_input['Difficulty'].map(difficulty_mapping)

    user_input_imputed = imputer.transform(user_input)
    user_scaled = scaler.transform(user_input_imputed)

    user_cluster = kmeans.predict(user_scaled)

    plt.figure(figsize=(10, 6))
    for cluster in selected_subject_df['Cluster'].unique():
        cluster_data = selected_subject_df[selected_subject_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=100, c='red',
                label='Centroids')
    plt.scatter(user_scaled[:, 0], user_scaled[:, 1], marker='*', s=100, c='green', label='User Input')
    plt.title(f'K-means Clustering of {subject} Courses on {platform} with User Input')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()

    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()

    return img_data, selected_subject_df[selected_subject_df['Cluster'] == user_cluster[0]]
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
questions = [
    ("How does exception handling work in Python, and what is the purpose of the `try`, `except`, and `finally` blocks?",
     ["Exception handling is used to terminate the program if an error occurs.",
      "`try` block is where the normal code execution occurs, `except` block handles exceptions, and `finally` block is optional and always executed.",
      "`try` block handles exceptions, `except` block is where the normal code execution occurs, and `finally` block is optional.",
      "`try` block handles exceptions, `except` block is optional, and `finally` block is executed only if an exception occurs."],
     3),
    ("What does SQL stand for?",
     ["Structured Question Language",
      "Structured Query Language",
      "Simple Question Language",
      "Simple Query Language"],
     4),
    ("Which SQL statement is used to update data in a database?",
     ["INSERT",
      "UPDATE",
      "MODIFY",
      "ALTER"],
     5),
    ("In SQL, what is the purpose of the GROUP BY clause?",
     ["To filter the results of a query",
      "To sort the results of a query",
      "To group rows that have the same values in specified columns",
      "To join multiple tables"],
     6),
    ("What is the purpose of a scatter plot in data visualization?",
     ["To show the distribution of a single variable",
      "To display the relationship between two continuous variables",
      "To compare different categories of data",
      "To show the proportion of each category in a dataset"],
     7),
    ("What is the main goal of exploratory data analysis (EDA) in data science?",
     ["To make predictions about future data",
      "To summarize the main characteristics of a dataset",
      "To design experiments for collecting new data",
      "To visualize data using charts and graphs"],
     8),
    ("If the ratio of boys to girls in a class is 3:5 and there are 24 boys, how many girls are there in the class?",
     ["30",
      "40",
      "45",
      "60"],
     9),
    ("A train travels at a speed of 50 km/h for the first half-hour and then at 60 km/h for the next half-hour. What is the average speed of the train for the entire journey?",
     ["54 km/h",
      "55 km/h",
      "56 km/h",
      "58 km/h"],
     10),
]


correct_answers = {
    1: 1,  # The correct answer for question 1 is the first option (index 1)
    2: 2,  # The correct answer for question 2 is the second option (index 2)
    3: 2,
    4: 3,
    5: 2,
    6: 3,
    7: 4,
    8: 2,    
}

def ask_question(question_data):
    question, options, correct_option = question_data
    return question, options, correct_option


@app.route("/", methods=['GET', 'POST'])
def index():
    img_data = None
    recommended_courses_df = None

    if request.method == 'POST':
        subject = request.form['subject']
        platform = request.form['platform']
        difficulty = request.form['difficulty']
        duration = int(request.form['duration'])
        rating = float(request.form['rating'])

        img_data, recommended_courses_df = run_kmeans(subject, platform, difficulty, duration, rating)

    return render_template('index.html', img_data=img_data, recommended_courses_df=recommended_courses_df)

#news
newsapi_api_key = "263f24e3d72e4880ab9ce9559725bef3"

# Community
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
db = SQLAlchemy(app)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    message = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()



@app.route("/Chatbot")
def home():
    return render_template("Chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/browse')
def browse():
    return render_template('browse.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/open_website')
def open_website():
    return redirect('https://college-nirf-rank-predictor.onrender.com/')


@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/get_it_market_news')
def get_it_market_news():
    newsapi = NewsApiClient(api_key=newsapi_api_key)

    try:
        it_market_news_data = newsapi.get_top_headlines(category='technology', language='en', country='us', page_size=50)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify(it_market_news_data)

@app.route('/community')
def community():
    messages = Message.query.all()
    return render_template('community.html', messages=messages)

@app.route('/chat', methods=['POST'])
def chat():
    username = request.form['username']
    message_text = request.form['message']
    message = Message(username=username, message=message_text)
    db.session.add(message)
    db.session.commit()
    return redirect(url_for('community'))

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/user_input', methods=['POST'])
def user_input():
    user_choice = request.form['choice']

    if user_choice == "1":
        return render_template('diploma.html')
    elif user_choice == "2":
        return render_template('eleventh.html')
    else:
        return render_template('error.html', message="Invalid input. Please enter a valid choice.")

@app.route('/specialization', methods=['POST'])
def specialization():
    specialization_choice = request.form['specialization_choice']

    if specialization_choice == "1":
        return "You selected Engineering (Diploma)."
    elif specialization_choice == "2":
        return "You selected Medical (Diploma)."
    else:
        return render_template('error.html', message="Invalid input. Please enter a valid choice.")

@app.route('/stream', methods=['POST'])
def stream():
    stream_choice = request.form['stream_choice']

    if stream_choice == "1":
        return render_template('science.html')
    elif stream_choice == "2":
        return render_template('arts.html')
    elif stream_choice == "3":
        return render_template('commerce.html')
    else:
        return render_template('error.html', message="Invalid input. Please enter a valid choice.")

@app.route('/career', methods=['POST'])
def career():
    career_choice = request.form['career_choice']

    if career_choice == "1":
        return "You selected Primary Teacher."
    elif career_choice == "2":
        return "You selected Artist."
    elif career_choice == "3":
        return "You selected Diploma."
    else:
        return render_template('error.html', message="Invalid input. Please enter a valid choice.")
    
@app.route('/quiz')
def quiz():
    total=0
    if request.method == 'POST':
        scores = {}

        for idx, question_data in enumerate(questions, start=1):
            question, options, correct_option = ask_question(question_data)

            user_answer = request.form.get(f'question_{idx}')
            user_answer = int(user_answer) if user_answer else None

            is_correct = user_answer == correct_option
            scores[f'question_{idx}'] = {'question': question, 'options': options, 'user_answer': user_answer, 'is_correct': is_correct}
            total += 1 if is_correct else 0
        return render_template('quiz.html', questions=questions, scores=scores)

    return render_template('quiz.html', questions=questions)




@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    # Process the form data here
    # Example:
    scores = {}
    total_score = 0
    total_questions = len(questions)

    for idx, question_data in enumerate(questions, start=1):
        question = question_data[0]
        options = question_data[1]
        user_answer = int(request.form.get(f'question_{idx}', 0))
        correct_option = correct_answers.get(idx, 0)
        is_correct = user_answer == correct_option

        scores[f'question_{idx}'] = {
            'question': question,
            'options': options,
            'user_answer': options[user_answer - 1] if user_answer else "No answer",
            'is_correct': is_correct
        }

        if is_correct:
            total_score += 1

    return render_template('quiz_result.html', scores=scores, total_score=total_score, total_questions=total_questions)



if __name__ == '__main__':
    app.run(debug=True)
