import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Charger rapidement une petite partie des données
df = pd.read_csv("annotated_emotions_cleaned.csv").head(10)

texts, labels = [], []
for _, row in df.iterrows():
    segment_texts = eval(row['Texts'])
    label = row['label']
    for text in segment_texts:
        texts.append(text.strip())
        labels.append(label)

# Encodage des labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Séparation rapide train-test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_encoded, test_size=0.5, random_state=42, stratify=labels_encoded
)

# Chargement du modèle HateBERT
model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Encodage des données
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateDataset(train_encodings, train_labels)
test_dataset = HateDataset(test_encodings, test_labels)

# Paramètres d'entraînement rapides avec logging précis
training_args = TrainingArguments(
    output_dir='./quick_results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=2,
    logging_steps=2,
    save_steps=2,
    learning_rate=2e-5,
    logging_dir='./logs',
)

# Trainer rapide avec compute_metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Entraînement rapide
trainer.train()

# Évaluation finale
eval_results = trainer.evaluate()
print(eval_results)

# Récupération des logs d'entraînement
log_history = trainer.state.log_history

# Graphiques interactifs des résultats
train_loss = [log['loss'] for log in log_history if 'loss' in log]
eval_steps = [log['step'] for log in log_history if 'eval_loss' in log]
eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
accuracy = [log['eval_accuracy'] for log in log_history if 'eval_accuracy' in log]
precision = [log['eval_precision'] for log in log_history if 'eval_precision' in log]
recall = [log['eval_recall'] for log in log_history if 'eval_recall' in log]
f1 = [log['eval_f1'] for log in log_history if 'eval_f1' in log]

fig, axs = plt.subplots(2, figsize=(10, 10))

# Loss
axs[0].plot(train_loss, label='Train Loss')
axs[0].plot(eval_steps, eval_loss, label='Eval Loss')
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss during Training')
axs[0].legend()

# Metrics
axs[1].plot(eval_steps, accuracy, label='Accuracy')
axs[1].plot(eval_steps, precision, label='Precision')
axs[1].plot(eval_steps, recall, label='Recall')
axs[1].plot(eval_steps, f1, label='F1-score')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Metric value')
axs[1].set_title('Evaluation Metrics')
axs[1].legend()

plt.tight_layout()
plt.show()
