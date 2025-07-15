from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# === Setup Flask ===
app = Flask(__name__)

# === Load data and train model (sử dụng đoạn code huấn luyện ban đầu) ===
import os
import kagglehub

df = kagglehub.dataset_download("basilb2s/language-detection")
df_path = os.path.join(df, 'Language Detection.csv')
df = pd.read_csv(df_path)
df.dropna(inplace=True)
df['Text'] = df['Text'].str.lower().str.strip()

X = df["Text"]
y = np.array(df["Language"])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

cv = CountVectorizer()
X_vect = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X_vect, y_encoded)

# === Route Trang chủ ===
@app.route('/')
def index():
    return render_template('index.html')

# === Route Dự đoán ===
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text_input'].lower().strip()
    input_vector = cv.transform([input_text])
    prediction = model.predict(input_vector)
    predicted_language = le.inverse_transform(prediction)[0]
    return render_template('result.html', input_text=input_text, predicted_language=predicted_language)

# === Chạy server ===
if __name__ == '__main__':
    app.run(debug=True)
