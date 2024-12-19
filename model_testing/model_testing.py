# Import packages

import os
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Argument parser
def parser():
    """
    The user can specify which finetuned model to test.
    The function will then parse command-line arguments and make them lower case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        "-m",
                        required = True,
                        choices = ["original", "neutral", "mix"],
                        help = "Specify dataset for finetuning (original, neutral, or mixed)")  
    args = parser.parse_args()
    args.model = args.model.lower()
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


def load_tokenizer(model):
    finetuned_model_path = f"/work/SofieNørboMosegaard#5741/NLP/NLP-exam/finetuned_models-OLD/BERT_finetuned_{model}_train"
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
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


def load_model(model):
    '''
    Load the finetuned model
    '''
    finetuned_model_path = f"/work/SofieNørboMosegaard#5741/NLP/NLP-exam/finetuned_models-OLD/BERT_finetuned_{model}_train"
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path)
    return model


def test_model(model, dataloader):

    model.eval()

    predictions = []
    true_labels = []

    for batch in dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask = b_input_mask)
            logits = outputs.logits

        # Convert logits to predicted class labels
        preds = torch.argmax(logits, dim = 1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(b_labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics, predictions


def save_results(metrics_female, metrics_male, predictions_female, predictions_male, metrics_path, predictions_path):
    with open(metrics_path, 'ab') as f:
        pickle.dump(metrics_female, f)
        pickle.dump(metrics_male, f)
    with open(predictions_path, 'ab') as f:
        pickle.dump(predictions_female, f)
        pickle.dump(predictions_male, f)



def calculate_bias(female_preds, male_preds):
    bias = np.mean(np.array(female_preds) - np.array(male_preds))
    return bias



# Main script

def main():
    
    args = parser()

    # Load data
    F_train_text, F_train_labels = load_data("/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/all_female_test.csv")
    M_train_text, M_train_labels = load_data("/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/all_male_test.csv")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Tokenize data
    F_input_ids, F_attention_masks = tokenize_data(F_train_text, tokenizer)
    M_input_ids, M_attention_masks = tokenize_data(M_train_text, tokenizer)


    # Create dataset
    F_dataset = TensorDataset(F_input_ids, F_attention_masks, F_labels)
    M_dataset = TensorDataset(M_input_ids, M_attention_masks, M_labels)

    # DataLoader
    batch_size = 16
    F_dataloader = DataLoader(F_dataset, batch_size = batch_size, shuffle = True)
    M_dataloader = DataLoader(M_dataset, batch_size = batch_size, shuffle = True)

    # Load fine-tuned omdel
    model = load_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Test!
    F_metrics, F_predictions = test_model(pretrained_model, F_dataloader)
    M_metrics, M_predictions = test_model(pretrained_model, M_dataloader)
    print("Testing complete!")

    # Define dicts and save results
    metrics_female = {'Model': "Pretrained Model (Female Dataset)",
                    'Metrics': F_metrics,
                    'Bias': pretrained_bias}
    metrics_male = {'Model': "Pretrained Model (Male Dataset)",
                    'Metrics': M_metrics,
                    'Bias': pretrained_bias}
    predictions_female = {'Model': "Pretrained Model (Female Dataset)",
                        'Predictions': F_predictions,
                        'True_Labels': F_train_labels}
    predictions_male = {'Model': "Pretrained Model (Male Dataset)",
                        'Predictions': M_predictions,
                        'True_Labels': M_train_labels}

    output_path = f"/work/SofieNørboMosegaard#5741/NLP/NLP-exam/results/"
    save_results(metrics_female,
                metrics_male,
                predictions_female,
                predictions_male, 
                f"{output_path}/model_results.pkl", 
                f"{output_path}/predictions.pkl")
    
    print("Results saved!")

    # Calculate bias
    bias = calculate_bias(F_predictions, M_predictions)
    print(f"Pretrained model bias (Male - Female): {bias}")

    with open(f"{output_path}/bias_{args.model}.txt", "w") as f:
        f.write(f"{args.model} model bias (Male - Female): {bias}")
    print("Bias measurement saved!")

if __name__ == "__main__":
    main()