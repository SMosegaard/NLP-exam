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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


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


def save_results(metrics_female, metrics_male, predictions_female, predictions_male, metrics_path, predictions_path):
    with open(metrics_path, 'ab') as f:
        pickle.dump(metrics_female, f)
        pickle.dump(metrics_male, f)
    with open(predictions_path, 'ab') as f:
        pickle.dump(predictions_female, f)
        pickle.dump(predictions_male, f)



def calculate_bias(male_preds, female_preds):
    '''
    The function will calculate the bias of each sample. Additionally, the total
    bias will be calculated, which is simply the mean of all biases. This measure
    also indicates will also measure the directional bias (e.g., preference for
    female or male). Finally, it will take the mean of absolute biases, which
    measures the magnitude of bias, regardless of direction.
    '''
    bias = np.mean(np.array(male_preds) - np.array(female_preds))
    total_bias = np.mean(biases)
    absolute_bias = np.mean(np.abs(biases))
    return biases, total_bias, absolute_bias



def test_significance(biases):
    '''
    Determine significance of the biases by employing the Wilcoxon Signed-Rank Test
    '''
    _, p_value = wilcoxon(biases, zero_method = "pratt")
    return p_value


def create_bias_table(model, total_biases, absolute_biases, p_values):
    data = {"Model": model,
            "Total Bias": total_biases,
            "Absolute Bias": absolute_biases,
            "p-value": p_values}
    return pd.DataFrame(data)


def plot_bias(biases, models, labels):
    x = range(len(models))
    plt.bar(x, biases, tick_label = models, color = ['blue' if b >= 0 else 'red' for b in biases])
    plt.xlabel('Models')
    plt.ylabel('Bias')
    plt.title('Bias per Model')
    plt.axhline(0, color = 'black', linewidth = 0.5)
    plt.show()


# Main script

def main():
    
    args = parser()

    # Paths for datasets
    dataset_paths = {"female": "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/all_female_test.csv",
                    "male": "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/all_male_test.csv"}

    # Initialize dicts for predictions and metrics
    predictions = {}
    metrics_data = {}

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
        predictions[gender] = preds

        # Save metrics and predictions for each dataset
        metrics_data[gender] = {'Model': f"Finetuned {args.data} model (all-{gender} dataset)",
                                'Metrics': metrics,
                                'Total_Bias': None,  # Placeholder
                                'Abs_Bias': None} 
        predictions[gender] = {'Model': f"Finetuned {args.data} model (all-{gender} dataset)",
                                'Predictions': preds,
                                'True_Labels': labels}
        
        print(f"Finished testing on all{gender} data")
    
    # Calculate bias
    biases, total_bias, absolute_bias = calculate_bias(predictions['male']['Predictions'], predictions['female']['Predictions'])
    metrics_data["female"]["Total_Bias"] = total_bias
    metrics_data["male"]["Total_Bias"] = total_bias
    metrics_data["female"]["Abs_Bias"] = absolute_bias
    metrics_data["male"]["Abs_Bias"] = absolute_bias

    p_value = test_significance(biases)
    data = create_bias_table(args.model, total_biases, absolute_biases, p_values)
    plot_bias(biases, models, labels)


    # Save results
    output_path = f"/work/SofieNørboMosegaard#5741/NLP/NLP-exam/results/"
    save_results(metrics_data["female"],
                metrics_data["male"],
                predictions["female"],
                predictions["male"], 
                f"{output_path}/model_results.pkl", 
                f"{output_path}/predictions.pkl")

    print("Results saved!")


if __name__ == "__main__":
    main()

