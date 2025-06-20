import pandas as pd
import json
import os

from datasets import Dataset, DatasetDict
from transformers import (
   BertForSequenceClassification,
   BertTokenizer,
   Trainer,
   TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime

dt_string = '_'.join(str(datetime.now()).split('.')[0].split())

num_epochs = 1
batch_size = 32

json_path_train = './data/gz_all.json'
json_path_test = './data/gz_101.json'

with open(json_path_train, 'r', encoding='utf8') as f:
    data_train = json.load(f)
data_train = [el for el in data_train if 'difficulty' in el.keys()]

with open(json_path_test, 'r', encoding='utf8') as f:
    data_test = json.load(f)
data_test = [el for el in data_test if 'difficulty' in el.keys()]

df_train = pd.DataFrame(data_train)
print(f"Train value counts: {df_train['difficulty'].value_counts()}")
print(f"Total train samples: {len(df_train)}")

df_test = pd.DataFrame(data_test)
print(f"Test value counts: {df_test['difficulty'].value_counts()}")
print(f"Total train samples: {len(df_test)}")

# Label mapping
label_map = {
    'molto_facile': 0,
    'facile': 1,
    'media': 2,
    'difficile': 3,
    'molto_difficile': 4,
}

# Split the dataset into train and test sets
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)


model_name = 'dbmdz/bert-base-italian-cased'
model_save_name = f"{model_name.split('/')[-1]}_{dt_string}"
print(f"model_save_name: {model_save_name}")


# Load Italian BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5
)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    input_texts = []
    for i in range(len(examples['difficulty'])):
        input_text = ' '.join([
            str(examples[col][i]) for col in examples.keys()
            if col != 'difficulty'
        ])
        input_texts.append(input_text)

    tokenized = tokenizer(
        input_texts,
        truncation=True,
        padding=True
    )

    tokenized['labels'] = [label_map[examples['difficulty'][i]] for i in range(len(examples['difficulty']))]
    return tokenized

# Apply preprocessing
dataset_train = dataset_train.map(preprocess_function, batched=True, batch_size=batch_size)
dataset_val = dataset_val.map(preprocess_function, batched=True, batch_size=batch_size)

# Combine datasets
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val
})

# Define compute_metrics function
def compute_metrics_acc(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }
def compute_metrics_mse(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    mse = mean_squared_error(labels, preds)
    return {
        'mse': mse,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    evaluation_strategy="epoch",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics_acc,
)

# Train model
trainer.train()

results = trainer.evaluate()

# Print metrics
print(f"Accuracy: {results['eval_accuracy']}")
print(f"Loss: {results['eval_loss']}")

# Confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
predictions = trainer.predict(dataset["val"])
preds = predictions.predictions.argmax(-1)
true_labels = dataset["val"]["labels"]

results_save_path = os.path.join('./results/val', model_save_name)
print(f"Saving classification report to: {results_save_path}")
print(classification_report(true_labels, preds), file = open(results_save_path, 'w', encoding='utf8'))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, preds))

# save model
model.save_pretrained(os.path.join('./models', model_save_name))
tokenizer.save_pretrained(os.path.join('./models', model_save_name))