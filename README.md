# AI-Powered Support Ticket Classification

## Overview
This project uses Natural Language Processing (NLP) and Machine Learning to predict the priority of customer support tickets from their subject and body text.

The model uses TF-IDF to convert ticket text into numerical features and Logistic Regression to classify ticket priority.

## Dataset
The script expects a CSV file named `Support_tickets.csv` in the project folder. It must contain these columns:

- `subject`
- `body`
- `priority`

The dataset is not included in this project bundle. Add your dataset locally before running the script, and make sure you have permission to share it before uploading it to a public GitHub repository.

## Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Setup

Install the required libraries:

```bash
pip install -r requirements.txt
```

Place `Support_tickets.csv` in this folder, then run:

```bash
python support_ticket_classifier.py
```

## Workflow
1. Load and inspect the dataset.
2. Combine ticket subjects and descriptions.
3. Clean ticket text.
4. Split data into training and testing sets.
5. Convert text to numerical features using TF-IDF.
6. Train a Logistic Regression classifier.
7. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Save the trained model and predict priorities for new tickets.

## Output files
Running the script creates:
- `support_ticket_model.pkl` — the trained model.
- `confusion_matrix.png` — the evaluation plot.

## Important
Model performance depends on the dataset, label quality, and class distribution. The model's predictions should support human review rather than replace it, especially for urgent or safety-critical customer issues.
