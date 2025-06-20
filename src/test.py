import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from datetime import datetime

# Paths
model_name = "./models/bert-base-italian-cased_2025-01-29_22:29:11"
test_data_path = "./data/gz_101.json"

# Load test data
with open(test_data_path, 'r', encoding='utf8') as f:
    data_test = json.load(f)
data_test = [el for el in data_test if 'difficulty' in el.keys()]

df_test = pd.DataFrame(data_test)
print(f"Test value counts: {df_test['difficulty'].value_counts()}")
print(f"Total test samples: {len(df_test)}")

# Label mapping
label_map = {
    'molto_facile': 0,
    'facile': 1,
    'media': 2,
    'difficile': 3,
    'molto_difficile': 4,
}

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 5)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

# Preprocessing function
def preprocess_function(examples):
    input_texts = [' '.join([str(examples[col][i]) for col in examples.keys() if col != 'difficulty']) for i in range(len(examples['difficulty']))]
    tokenized = tokenizer(input_texts, truncation=True, padding=True, return_tensors='pt')
    labels = torch.tensor([label_map[examples['difficulty'][i]] for i in range(len(examples['difficulty']))])
    return tokenized, labels

# Preprocess test data
test_dataset = Dataset.from_pandas(df_test)
tokenized_inputs, true_labels = preprocess_function(df_test)

data_loader = DataLoader(TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], true_labels), batch_size=8)

# Make predictions
preds = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend(torch.argmax(outputs.logits, dim=-1).tolist())
        true_labels.extend(labels.tolist())

dt_string = '_'.join(str(datetime.now()).split('.')[0].split())
model_save_name = f"{model_name.split('/')[-1]}"

test_acc = accuracy_score(true_labels, preds)
print(f"Test Accuracy: {test_acc:.4f}")

results_save_path = os.path.join('./results/test', model_save_name)
print(f"Saving test classification report to: {results_save_path}")
print(classification_report(true_labels, preds))
print(classification_report(true_labels, preds), file = open(results_save_path, 'w', encoding='utf8'))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, preds))