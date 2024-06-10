import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
import json
import random
from flask import Flask, render_template, request

lemmatizer = WordNetLemmatizer()

# Load dữ liệu tin tức
with open('news_scrap_data.json', encoding='utf-8') as file:
    news_scrap_data = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

import random

def chatbot_response(msg):
    categories = ["Thời sự", "Góc nhìn", "Thế giới", "Podcasts", "Kinh doanh", "Bất động sản", "Khoa học", 
                  "Giải trí", "Thể thao", "Pháp luật", "Giáo dục", "Sức khỏe", "Đời sống", "Du lịch", 
                  "Số hóa", "Xe", "Ý kiến", "Tâm sự", "Thư giãn"]
    
    selected_category = None
    for category in categories:
        if category.lower() in msg.lower():
            selected_category = category
            break
    
    news_responses = []
    
    if selected_category:
        # Lọc ra tất cả tin tức trong danh mục được chọn
        selected_news = [news for news in news_scrap_data['intents'] if news['category'] == selected_category]
        
        # Kiểm tra xem có tin tức nào trong danh mục không
        if selected_news:
            # Chọn một tin tức ngẫu nhiên từ danh sách tin tức đã lọc
            random_news = random.choice(selected_news)
            
            title = random_news['title']
            summary = random_news['summary']
            link = random_news['news_link']
            
            news_responses.append(f"<strong>Tiêu đề:</strong> {title}<br> \n <strong>Nội dung:</strong> {summary}<br> \n <strong>Chủ đề:</strong> {random_news['category']}<br> \n <a href='{link}' target='_blank'>Xem thêm</a><br><br>")
        else:
            news_responses.append("Không có tin tức nào trong chủ đề này.")
            
        return "".join(news_responses)

    elif 'tin tức' in msg.lower():
        random.shuffle(news_scrap_data['intents'])  # Shuffle the news list
        news = news_scrap_data['intents'][0]
        title = news['title']
        summary = news['summary']
        link = news['news_link']
        return f"<strong>Tiêu đề:</strong> {title}<br> \n <strong>Nội dung:</strong> {summary}<br> \n <strong>Chủ đề:</strong> {news['category']}<br> \n <a href='{link}' target='_blank'>Xem thêm</a><br><br>"


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(debug=True)
