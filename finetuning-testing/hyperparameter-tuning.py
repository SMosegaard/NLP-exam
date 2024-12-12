
import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from torch.optim import lr_scheduler
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
from collections import Counter
from tqdm import tqdm
from deep_translator import GoogleTranslator
import seaborn as sns
import matplotlib.pyplot as plt
import optuna


# tuning test


def objective(trial):

    # Hyperparameters to tune
    dropout_rate = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    num_epochs = trial.suggest_int('epochs', 2, 15)
    lr = trial.suggest_loguniform('lr', 5e-5, 1e-3)
    warmup = trial.suggest_categorical('warmup', [0.0, 0.1])

    # load data
    train_file = "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/original_train.csv"
    train_text, train_labels = load_data(train_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained("vesteinn/DanskBERT", num_labels = 2)
    model.classifier = nn.Sequential(nn.Dropout(dropout_rate), model.classifier)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize and prepare data
    input_ids, attention_masks, labels = tokenize_data(train_text, train_labels, tokenizer)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataset, val_dataset = train_test_split(dataset, test_size = 0.3, stratify = labels)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_dataloader = DataLoader(val_dataset, batch_size = batch_size)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8, weight_decay = 0.01)
    total_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(warmup * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = total_steps)


    # Training Loop

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")

        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc = "Training"):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()

            result = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
            loss = result.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        print("Running Validation...")
        model.eval()
        total_val_loss = 0

        for batch in tqdm(validation_dataloader, desc = "Validation"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                result = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
                loss = result.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        trial.report(avg_val_loss, epoch)

        # suggest whether to end the current hyperparameter trail (works like early stopping)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    save_trial_results(trial.number, trial.params, best_val_loss)
    return best_val_loss



# Function to compute the accuracy of predictions vs labels
def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Sentiment'] = df['Sentiment'].map({'pos': 1, 'neg': 0})
    text = df['text'].tolist()
    labels = df['Sentiment'].tolist()
    return text, labels


def tokenize_data(train_text, labels, tokenizer):
    input_ids = []
    attention_masks = []
    for text in train_text:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = 512,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
    # Convert to tensors
    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

if __name__ == "__main__":

    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 10)

    print("Best Hyperparameters:", study.best_params)