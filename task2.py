import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_data(folder):
    texts = []
    labels = []
    for label, subfolder in [(1, 'pos'), (0, 'neg')]:
        subfolder_path = os.path.join(folder, subfolder)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

# Load training data
train_folder = r"E:\aclImdb\train"
train_texts, train_labels = load_data(train_folder)

test_folder = r"E:\aclImdb\test"
test_texts, test_labels = load_data(test_folder)

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
y_train = train_labels
y_test = test_labels

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
