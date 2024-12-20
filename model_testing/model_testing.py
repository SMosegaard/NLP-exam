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
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
                        help = "Specify dataset for finetuning (original, neutral, or mix)")  
    args = parser.parse_args()
    args.model = args.model.lower()
    return args


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


def test_model(model, dataloader, device):

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



def save_results_to_csv(metrics_data, predictions, output_path):
    metrics_list = []
    for gender, models in metrics_data.items():
        for model_name, model_data in models.items():
            metrics_list.append({'Gender': gender,
                                'Model': model_data['Model'],
                                'Accuracy': model_data['Metrics']['accuracy'],
                                'Precision': model_data['Metrics']['precision'],
                                'Recall': model_data['Metrics']['recall'],
                                'F1 Score': model_data['Metrics']['f1_score']})
    metrics_df = pd.DataFrame(metrics_list)

    predictions_list = []
    for gender, models in predictions.items():
        for model_name, model_data in models.items():
            predictions_list.append({'Gender': gender,
                                    'Model': model_data['Model'],
                                    'Predictions': model_data['Predictions'],
                                    'True Labels': model_data['True_Labels']})    
    predictions_df = pd.DataFrame(predictions_list)

    metrics_df.to_csv(f"{output_path}/model_metrics.csv", mode = 'a', index = False)
    predictions_df.to_csv(f"{output_path}/model_predictions.csv", mode = 'a', index = False)

    print(f"Results saved as CSV at {output_path}")


def save_results(metrics_female, metrics_male, predictions_female, predictions_male, metrics_path, predictions_path):
    with open(metrics_path, 'ab') as f:
        pickle.dump(metrics_female, f)
        pickle.dump(metrics_male, f)
    with open(predictions_path, 'ab') as f:
        pickle.dump(predictions_female, f)
        pickle.dump(predictions_male, f)



# Main script

def main():
    
    args = parser()

    # Paths for datasets
    dataset_paths = {"female": "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/all_female_test.csv",
                    "male": "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/all_male_test.csv"}

    # Initialize dicts for predictions and metric
    metrics_data = {"female": {}, "male": {}}
    predictions = {"female": {}, "male": {}}

    # Loop through each test set
    for gender, file_path in dataset_paths.items():
        print(f"Processing {gender} dataset")

        # Load data
        text, labels = load_data(file_path)

        # Load tokenizer
        tokenizer = load_tokenizer(args.model)

        # Tokenize data
        input_ids, attention_masks, label_tensors = tokenize_data(text, labels, tokenizer)

        # Create dataset
        dataset = TensorDataset(input_ids, attention_masks, label_tensors)

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

        # Load fine-tuned model
        model = load_model(args.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Test model on the dataset
        metrics, preds = test_model(model, dataloader, device)
        
        #predictions[gender] = preds

        # Initialize gender-specific dictionaries in predictions if not already done
        if gender not in predictions:
            predictions[gender] = {}

        # Save metrics and predictions for each dataset
        metrics_data[gender][args.model] = {'Model': f"Finetuned {args.model} model (all-{gender} dataset)",
                                            'Model': args.model,
                                            'Gender': gender,
                                            'Metrics': metrics} 
        predictions[gender][args.model] = {'Model': f"Finetuned {args.model} model (all-{gender} dataset)",
                                            'Model': args.model,
                                            'Gender': gender,
                                            'Predictions': preds,
                                            'True_Labels': labels}
        
        print(f"Finished testing on all-{gender} data")

    # Save results
    output_path = f"/work/SofieNørboMosegaard#5741/NLP/NLP-exam/results/"
    save_results_to_csv(metrics_data, predictions, output_path)

    print("Results saved!")


if __name__ == "__main__":
    main()

