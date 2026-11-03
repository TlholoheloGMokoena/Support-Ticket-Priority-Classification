# =====================================================
# AI-POWERED SUPPORT TICKET CLASSIFICATION SYSTEM
# =====================================================

import pandas as pd
import re
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. LOAD DATASET
# Place Support_tickets.csv in the same folder as this script.
df = pd.read_csv("Support_tickets.csv")
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. CHECK REQUIRED COLUMNS
required_columns = ["subject", "body", "priority"]
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Dataset is missing the required column: {column}")

# 3. PREPARE DATA
df = df.dropna(subset=["priority"]).copy()
df["text"] = df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
df = df[df["text"].str.strip() != ""].copy()

X = df["text"]
y = df["priority"].astype(str).str.strip().str.lower()

valid_rows = y != ""
X = X[valid_rows]
y = y[valid_rows]

print("\nPriority distribution:")
print(y.value_counts())

# 4. TEXT CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. BUILD AND TRAIN MODEL
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 1),
        max_features=10000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# 7. PREDICT AND EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# 8. CONFUSION MATRIX
labels = model.classes_
cm = confusion_matrix(y_test, predictions, labels=labels)

print("\nConfusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot(cmap="Blues", values_format="d")
plt.title("Support Ticket Priority Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 9. SAVE TRAINED MODEL
joblib.dump(model, "support_ticket_model.pkl")
print("\nTrained model saved as support_ticket_model.pkl")

# 10. TEST NEW SUPPORT TICKETS
new_tickets = [
    "Server outage affecting all users. Customers cannot access the platform.",
    "I have a question about my monthly billing statement."
]

print("\nNEW TICKET PREDICTIONS")
for ticket in new_tickets:
    prediction = model.predict([ticket])[0]
    print("\nTicket:", ticket)
    print("Predicted priority:", prediction)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
