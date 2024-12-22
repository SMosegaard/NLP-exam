# Import packages

import os
import sys
import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from torch.optim import lr_scheduler
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
from tqdm import tqdm
import optuna


# Define functions

def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)


def objective(trial, data, train_dataset, val_dataset):

    # Hyperparameters to tune
    dropout_rate = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    num_epochs = trial.suggest_int('epochs', 2, 15) 
    lr = trial.suggest_loguniform('lr', 5e-5, 1e-3)
    warmup = trial.suggest_categorical('warmup', [0.0, 0.1])

    # Prepare data
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_dataloader = DataLoader(val_dataset, batch_size = batch_size)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained("vesteinn/DanskBERT", num_labels = 2)
    model.classifier = nn.Sequential(nn.Dropout(dropout_rate), model.classifier)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    save_trial_results(trial.number, trial.params, best_val_loss, warmup, data)
    return best_val_loss


def save_trial_results(trial_number, params, best_val_loss, warmup, data):

    params['warmup'] = params.get('warmup', None)  # Ensure warmup exists in case of issues

    result_dict = {
        'trial': trial_number,
        'params': params,
        'best_val_loss': best_val_loss,
        'warmup': warmup
    }
    df = pd.DataFrame([result_dict])    

    results_path = f'/finetuned_models/BERT_finetuned_{data}_train/optuna_results_{data}.csv'
    file_exists = os.path.exists(results_path)    
    df.to_csv(results_path, mode = 'a', header = not file_exists, index = False)


def main(data, train_dataset, val_dataset):

    study = optuna.create_study(direction = "minimize")
    study.optimize(lambda trial: objective(trial, data, train_dataset, val_dataset), n_trials = 10)

    print("Best Hyperparameters:", study.best_params)

    return study.best_params

if __name__ == "__main__":
    main()
 