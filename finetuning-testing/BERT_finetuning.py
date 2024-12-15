# Import packages

import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import shuffle
from tqdm import tqdm
from deep_translator import GoogleTranslator
import seaborn as sns
import matplotlib.pyplot as plt
from hyperparameter_tuning import main as hyperparameter_tuning_main 



# Argument parser
def parser():
    """
    The user can specify which dataset to finetune the model on and/or to perform hyperparameter
    when executing the script. The function will then parse command-line arguments and make them lower case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        "-d",
                        required = True,
                        choices = ["original_train", "neutral_train", "mix_train"],
                        help = "Specify dataset for finetuning (original, neutral, or mixed gender)")  
    parser.add_argument("--hyperparameter_tuning",
                        "-ht",
                        required = True,
                        default = "no",
                        help = "Perform hyperparameter tuning (yes or no)")
      
    args = parser.parse_args()
    args.data = args.data.lower()
    args.HyperparameterTuning = args.HyperparameterTuning.lower()
    return args


# Defined functions
def load_data(file_path):
    """
    Loads data and converts Sentiment classes to numeric labels (1 for positive, 0 for negative).
    """
    df = pd.read_csv(file_path)
    df['Sentiment'] = df['Sentiment'].map({'pos': 1, 'neg': 0})
    text = df['text'].tolist()
    labels = df['Sentiment'].tolist()
    return text, labels


def train_val_split(text, labels):
    '''
    Create a 70-30 train-validation split
    '''
    train_text, val_text, train_labels, val_labels = train_test_split(text, labels,
                                                                    test_size = 0.3,
                                                                    stratify = labels,
                                                                    random_state = 123)
    return train_text, val_text, train_labels, val_labels


def back_translate(text, src_lang = 'da'):
    '''
    Data augmentation using back-translation. Due to API limit of 5000 characters,
    the function checks the lenght of the input text.
    '''
    if len(text) > 5000:
        print(f"Text length is {len(text)} characters. Skipping translation")
        return None
    lang_list = ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'sv', 'no', 'fi']
    trans_lang = random.choice(lang_list)
    try:
        translated_text = GoogleTranslator(source = src_lang, target = trans_lang).translate(text)
        back_translated_text = GoogleTranslator(source = trans_lang, target = src_lang).translate(translated_text)
        return back_translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


def random_deletion(text, p = 0.3):
    '''
    Data augmentation using random deletion.
    '''
    words = text.split()
    if len(words) == 1:
        return words
    remaining = [word for word in words if random.random() > p]
    if len(remaining) == 0:
        return random.choice(words) 
    else:
        return ' '.join(remaining)


def random_swap(text, n = 5):
    '''
    Data augmentation using random swap
    '''
    words = text.split()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


def augment_data(train_text, train_labels):
    '''
    Data augmentation (combine the three functions of different augmentation methods)
    '''
    augmented_text = []
    augmented_labels = []
    for text, label in zip(train_text, train_labels):
        back_translated_text = back_translate(text, src_lang = 'da')
        if back_translated_text: 
            augmented_text.append(back_translated_text)
            augmented_labels.append(label)
        augmented_text.append(random_deletion(text))
        augmented_labels.append(label)
        augmented_text.append(random_swap(text, 5))
        augmented_labels.append(label)
    return augmented_text, augmented_labels


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")
    return tokenizer


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


def load_model(dropout):
    '''
    Load the Danish BERT model and add a dropout layer
    '''
    model = AutoModelForSequenceClassification.from_pretrained("vesteinn/DanskBERT", num_labels = 2)
    model.classifier = nn.Sequential(nn.Dropout(dropout), model.classifier)
    return model


def define_optimizer(model, lr):
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8, weight_decay = 0.01)
    return optimizer


def define_scheduler(train_dataloader, epochs, warmup, optimizer):
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(warmup * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = num_warmup_steps,
                                                num_training_steps = total_steps)
    return scheduler


def flat_accuracy(preds, labels):
    '''
    Compute model accuracy
    '''
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)


def compute_metrics(preds, labels):
    '''
    Calculate F1, recall, and precision
    '''
    preds_flat = preds.argmax(axis = 1).flatten()
    labels_flat = labels.flatten()
    f1 = f1_score(labels_flat, preds_flat, average = 'weighted')
    recall = recall_score(labels_flat, preds_flat, average = 'weighted')
    precision = precision_score(labels_flat, preds_flat, average = 'weighted')
    return f1, recall, precision


def train_epoch(model, train_dataloader, optimizer, scheduler, device):

    model.train()

    total_train_loss = 0
    total_train_accuracy = 0
    total_train_f1 = 0
    total_train_recall = 0
    total_train_precision = 0

    train_iterator = tqdm(enumerate(train_dataloader), total = len(train_dataloader), desc = "Training")
    for step, batch in train_iterator:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        # Forward pass
        result = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
        loss = result.loss
        logits = result.logits

        # Move logits and labels to CPU
        preds = logits.detach().cpu().numpy()
        labels = b_labels.cpu().numpy()

        # Calculate metrics
        total_train_loss += loss.item()
        total_train_accuracy += flat_accuracy(preds, labels)
        f1, recall, precision = compute_metrics(preds, labels)
        total_train_f1 += f1
        total_train_recall += recall
        total_train_precision += precision
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    # Calculate averages
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    avg_train_f1 = total_train_f1 / len(train_dataloader)
    avg_train_recall = total_train_recall / len(train_dataloader)
    avg_train_precision = total_train_precision / len(train_dataloader)
    current_lr = scheduler.get_last_lr()[0]

    return avg_train_loss, avg_train_accuracy, avg_train_f1, avg_train_recall, avg_train_precision, current_lr


def evaluate_epoch(model, validation_dataloader, device):
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    total_eval_f1 = 0
    total_eval_recall = 0
    total_eval_precision = 0

    validation_iterator = tqdm(validation_dataloader, desc = "Validation")
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
            loss = result.loss
            logits = result.logits

            # Move logits and labels to CPU
            preds = logits.detach().cpu().numpy()
            labels = b_labels.cpu().numpy()

            # Calculate metrics
            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(preds, labels)
            f1, recall, precision = compute_metrics(preds, labels)
            total_eval_f1 += f1
            total_eval_recall += recall
            total_eval_precision += precision

    # Calculate averages
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_f1 = total_eval_f1 / len(validation_dataloader)
    avg_val_recall = total_eval_recall / len(validation_dataloader)
    avg_val_precision = total_eval_precision / len(validation_dataloader)

    return avg_val_loss, avg_val_accuracy, avg_val_f1, avg_val_recall, avg_val_precision


def save_training_stats(training_stats, output_path):
    df_stats = pd.DataFrame(training_stats).set_index('epoch')
    stats_file = os.path.join(output_path, "training_stats.csv")
    df_stats.to_csv(stats_file)
    return df_stats



def plot_training_metrics(df_stats, output_path):
    """
    Plot training and validation loss, accuracy, and F1 score from a dataframe.
    Save plot to specified output path
    """
    sns.set(style = 'darkgrid')
    sns.set(font_scale = 1.5)
    plt.rcParams["figure.figsize"] = (18, 6)
    fig, axes = plt.subplots(1, 3)

    # Loss
    axes[0].plot(df_stats['Training loss'], 'b-o', label = "Training")
    axes[0].plot(df_stats['Validation loss'], 'g-o', label = "Validation")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_xticks(range(len(df_stats['Training loss'])))
    axes[0].set_xticklabels(range(1, len(df_stats['Training loss']) + 1))

    # Accuracy
    axes[1].plot(df_stats['Training accuracy'], 'b-o', label = "Training")
    axes[1].plot(df_stats['Validation accuracy'], 'g-o', label = "Validation")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_xticks(range(len(df_stats['Training accuracy'])))
    axes[1].set_xticklabels(range(1, len(df_stats['Training accuracy']) + 1))

    # F1 Score
    axes[2].plot(df_stats['Training F1'], 'b-o', label = "Training")
    axes[2].plot(df_stats['Validation F1'], 'g-o', label = "Validation")
    axes[2].set_title("Training & Validation F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].set_xticks(range(len(df_stats['Training F1'])))
    axes[2].set_xticklabels(range(1, len(df_stats['Training F1']) + 1))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()



class EarlyStopping:
    """
    Stops training when the validation accuracy does not improve after a given patience

    patience: Number of epochs to wait before stopping if no improvement.
    delta: Minimum change in the monitored accuracy to qualify as an improvement.
    """
    def __init__(self, patience = 5, delta = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_score is None or val_loss > self.best_score + self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Main script

def main():
    
    args = parser()

    # Load data
    file_path = f'/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/{args.data}.csv'
    text, labels = load_data(file_path)
    train_text, val_text, train_labels, val_labels = train_val_split(text, labels)

    # Data augmentation due to limited data (apply to some of the training data)
    train_text_250 = train_text[:250]
    train_labels_250 = train_labels[:250]
    augmented_text, augmented_labels = augment_data(train_text_250, train_labels_250)
    train_text = train_text + augmented_text
    train_labels = train_labels + augmented_labels
    train_text, train_labels = shuffle(np.array(train_text), np.array(train_labels),random_state = 123)
    train_text = train_text.tolist()
    train_labels = train_labels.tolist()

    # Tokenize data
    tokenizer = load_tokenizer()
    train_input_ids, train_attention_masks, train_labels = tokenize_data(train_text, train_labels, tokenizer)
    val_input_ids, val_attention_masks, val_labels = tokenize_data(val_text, val_labels, tokenizer)

    # Create dataset
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    # Default parameters dict
    parameters = {"lr": 5e-5,
                "batch_size": 16,
                "dropout": 0.5,
                "epochs": 15,
                "warmup": 0.1}

    # Perform hyperparameter tuning or simply use default parameters
    if args.HyperparameterTuning == 'yes':
        tuned_parameters = hyperparameter_tuning_main(args.data)
        print(f"Best Hyperparameters: {tuned_parameters}")
        parameters.update(tuned_parameters)

    # DataLoader
    batch_size = batch_size
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_dataloader = DataLoader(val_dataset, batch_size = batch_size)

    # Load model
    model = load_model(dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = define_optimizer(model, lr)
    scheduler = define_scheduler(train_dataloader, epochs, warmup, optimizer)
    
    #  Training Loop
    early_stopping = EarlyStopping(patience = 3, delta = 0.001)
    training_stats = []

    print("Starting training...")
    for epoch_i in range(epochs):
        print(f"Epoch {epoch_i + 1} / {epochs}")

        # Training
        avg_train_loss, avg_train_accuracy, avg_train_f1, avg_train_recall, avg_train_precision, current_lr = train_epoch(
            model, train_dataloader, optimizer, scheduler, device)

        print(f"Average training loss: {avg_train_loss}")
        print(f"Average training accuracy: {avg_train_accuracy}")
        print(f"Average training F1: {avg_train_f1}, recall: {avg_train_recall}, precision: {avg_train_precision}")
        print(f"Learning rate at the end of epoch {epoch_i + 1}: {current_lr:.8f}")

        # Validation
        print("Running Validation...")
        avg_val_loss, avg_val_accuracy, avg_val_f1, avg_val_recall, avg_val_precision = evaluate_epoch(
            model, validation_dataloader, device)
        
        print(f"Average validation loss: {avg_val_loss}")
        print(f"Average validation accuracy: {avg_val_accuracy}")
        print(f"Validation F1: {avg_val_f1}, recall: {avg_val_recall}, precision: {avg_val_precision}")

        # Save all stats
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training loss': avg_train_loss,
            'Training accuracy': avg_train_accuracy,
            'Training F1': avg_train_f1,
            'Training recall': avg_train_recall,
            'Training precision': avg_train_precision,
            'Validation loss': avg_val_loss,
            'Validation accuracy': avg_val_accuracy,
            'Validation F1': avg_val_f1,
            'Validation recall': avg_val_recall,
            'Validation precision': avg_val_precision,
        })

        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Training complete!")

    # Save fine-tuned model
    output_path = f"/work/SofieNørboMosegaard#5741/NLP/NLP-exam/finetuned_models/BERT_finetuned_{data}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model.save_pretrained(output_path) 
    tokenizer.save_pretrained(output_path)

    # Save finetuning stats
    df_stats = save_training_stats(training_stats, output_path)

    # Plot loss, accuracy, anf F1
    plot_training_metrics(df_stats, output_path)
    
    print(f"Model, metrics table, and metrics plot are saved to {output_path}")

if __name__ == "__main__":
    main()