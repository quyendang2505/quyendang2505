import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu
data = pd.read_csv("IMDB Dataset.csv")

# Chuyển đổi nhãn cảm xúc sang số (1: positive, 0: negative)
data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Làm sạch văn bản
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Loại bỏ URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Chỉ giữ lại chữ cái
    text = text.lower()  # Chuyển thành chữ thường
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Làm sạch dữ liệu
data['cleaned_text'] = data['review'].apply(clean_text)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'], data['label'], test_size=0.2, random_state=42
)

# Trích xuất đặc trưng bằng TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Sử dụng 5000 đặc trưng quan trọng nhất
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_tfidf)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Ma trận nhầm lẫn
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Dự đoán trên một số văn bản mới
sample_reviews = [
    "The movie was fantastic! I really enjoyed it.",
    "Absolutely terrible. Waste of time and money.",
    "It was okay, not the best but not the worst either."
]

# Làm sạch và trích xuất đặc trưng từ văn bản mới
sample_reviews_cleaned = [clean_text(review) for review in sample_reviews]
sample_reviews_tfidf = tfidf_vectorizer.transform(sample_reviews_cleaned)

# Dự đoán
sample_predictions = model.predict(sample_reviews_tfidf)
for review, sentiment in zip(sample_reviews, sample_predictions):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {'positive' if sentiment == 1 else 'negative'}")
    print("-" * 50)
