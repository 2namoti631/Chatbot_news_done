import nltk
import numpy as np
import joblib
import json
import random
from flask import Flask, render_template, request
from mtranslate import translate

# Load mô hình phân loại câu hỏi
model_detec = joblib.load('model_intent_detection.joblib')
vectorizer_detec = joblib.load('vectorizer_detec.joblib')
with open('label_question_detect.json', 'r', encoding='utf-8') as file:
    label_data = json.load(file)


# Load mô hình hội thoại  
model_conver = joblib.load('model_conversation.joblib')
vectorizer_conver = joblib.load('vectorizer_conver.joblib')


# Load dữ liệu tin tức
with open('news_scrap_data.json', encoding='utf-8') as file:
    news_scrap_data = json.load(file)

# Load dữ liệu intents từ file data.json
with open('data_basic_conver.json', encoding='utf-8') as file:
    intents = json.load(file)

def classify_intent_questions(msg):
    msg = translate(msg, "en")
    data_processing = vectorizer_detec.transform([msg]).toarray()
    predict = model_detec.predict(data_processing)
    x = predict[0]
    result = get_definition_and_trans(x)
    if result:
        return f" <strong>Chủ đề:</strong> {result['translation']}<br> \n <strong>Thuộc chủ đề lớn: </strong> {result['definition']}<br>"
    return "Không thể xác định chủ đề."

def get_definition_and_trans(word):
    result = {}
    for category, values in label_data.items():
        for subcategory, translation in values['types'].items():
            if subcategory == word:
                result['translation'] = translation
                result['definition'] = values['definition']
                return result
    return None

def classify_intent(msg):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in msg.lower():
                return intent
    return None
 

def chatbot_response(msg):
    categories = ["Thời sự", "Góc nhìn", "Thế giới", "Podcasts", "Kinh doanh", "Bất động sản", "Khoa học", 
                  "Giải trí", "Thể thao", "Pháp luật", "Giáo dục", "Sức khỏe", "Đời sống", "Du lịch", 
                  "Số hóa", "Xe", "Ý kiến", "Tâm sự", "Thư giãn"]

    intent = classify_intent(msg)
    if intent:
        return random.choice(intent['responses'])
    
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
    
    # code xác định chủ đề câu hỏi
    elif 'chủ đề của câu hỏi:' in msg.lower():
        # Tách đoạn văn bản khỏi câu hỏi
        content = msg.lower().replace('chủ đề của câu hỏi:', '').strip()
        if content:
            return classify_intent_questions(content)
        return "Vui lòng nhập đoạn văn bản cần phân loại."
      
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

