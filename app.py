import string
import random
import json
import csv
import joblib
import numpy as np
from flask import Flask, render_template, request
from mtranslate import translate

# Load mô hình và vector hóa của phân loại câu hỏi
model_detec = joblib.load('model_intent_detection.joblib')
vectorizer_detec = joblib.load('vectorizer_detec.joblib')

# load file data dữ liệu phân loại câu hỏi
with open('label_question_detect.json', 'r', encoding='utf-8') as file:
    label_data = json.load(file)

# Load mô hình và vector hóa của việc training hội thoại  
model_conver = joblib.load('model_conversation.joblib')
vectorizer_conver = joblib.load('vectorizer_conver.joblib')

# Load dữ liệu tin tức đã được scrapt về
with open('news_scrap_data.json', encoding='utf-8') as file:
    news_scrap_data = json.load(file)

# Load dữ liệu intents từ file hội thoại cơ bản cho chatbot
with open('data_basic_conver.json', encoding='utf-8') as file:
    intents = json.load(file)


# Chức năng đọc dữ liệu từ file csv của data hội thoại và các bước tiền xử lý để để có thể truy cập data
def load_data(filename):
    data = {}
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tag = row['Tag']
            if tag not in data:
                data[tag] = []
            data[tag].append({
                'Pattern': row['Pattern'].translate(str.maketrans('', '', string.punctuation)).lower(),
                'Response': row['Response']
            })
    return data

data_conversation = load_data('data_conver.csv')

# Chức năng phân loại chủ đề câu hỏi
def classify_intent_questions(msg):
    msg = translate(msg, "en")
    data_processing = vectorizer_detec.transform([msg]).toarray()
    predict = model_detec.predict(data_processing)
    x = predict[0]
    result = get_definition_and_trans(x)
    if result:
        return f"<strong>Chủ đề:</strong> {result['translation']}<br> \n <strong>Thuộc chủ đề lớn: </strong> {result['definition']}<br>"
    return "Không thể xác định chủ đề."

# chức năng để truy cập vào file json và lấy các biến liên quan đến câu hỏi được dự đoán để in ra. (Phân loại câu hỏi)
def get_definition_and_trans(word):
    result = {}
    for category, values in label_data.items():
        for subcategory, translation in values['types'].items():
            if subcategory == word:
                result['translation'] = translation
                result['definition'] = values['definition']
                return result
    return None

# Chức năng trả lời câu hỏi hội thoại theo model training
def get_response_from_model(msg, model_conver, vectorizer_conver, intents):
    bow = vectorizer_conver.transform([msg])
    predicted_tag = model_conver.predict(bow)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "Xin lỗi, tôi không hiểu câu hỏi của bạn."

#  Tổng hợp các chức năng và để chatbot hiện thị với người dùng
def chatbot_response(msg, data_conversation):
    
    # In ra Chủ đề của câu hỏi người dùng nhập vào
    if 'chủ đề của câu hỏi:' in msg.lower():
        # Tách đoạn văn bản khỏi câu hỏi
        content = msg.lower().replace('chủ đề của câu hỏi:', '').strip()
        if content:
            return classify_intent_questions(content)
        return "Vui lòng nhập đoạn văn bản cần phân loại."
      

    # In ra tin tức theo chủ đề
    categories = [
        "Thời sự", "Góc nhìn", "Thế giới", "Podcasts", "Kinh doanh", "Bất động sản", "Khoa học", 
                  "Giải trí", "Thể thao", "Pháp luật", "Giáo dục", "Sức khỏe", "Đời sống", "Du lịch", 
        "Số hóa", "Xe", "Ý kiến", "Tâm sự", "Thư giãn"
    ]
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

    # In ra tin tức chung 
    elif 'tin tức' in msg.lower():
        random.shuffle(news_scrap_data['intents'])  # Shuffle the news list
        news = news_scrap_data['intents'][0]
        title = news['title']
        summary = news['summary']
        link = news['news_link']
        return f"<strong>Tiêu đề:</strong> {title}<br> \n <strong>Nội dung:</strong> {summary}<br> \n <strong>Chủ đề:</strong> {news['category']}<br> \n <a href='{link}' target='_blank'>Xem thêm</a><br><br>"
    
    # Tìm input rồi in hội thoại tương ứng ra.
    for tag, patterns in data_conversation.items():
        for pattern in patterns:
            if pattern['Pattern'] == msg:
                return pattern['Response']

    # Nếu không thấy hội thoại i hệt và tương ứng thì sẽ in ra hội thoại được training
    return get_response_from_model(msg, model_conver, vectorizer_conver, intents)

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText, data_conversation)

if __name__ == "__main__":
    app.run(debug=True)
