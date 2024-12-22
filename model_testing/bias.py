import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def convert_predictions(predictions):
    '''
    As the predictions are saved in a np.int64 format, we need to convert them to integers
    '''
    return [int(value) for value in predictions]


def calculate_bias(male_preds, female_preds):
    '''
    The function will calculate the bias of each sample as well as the total bias
    (mean of all biases). The total bias measure will also indicates the directional
    bias (e.g., preference for female or male). Further, it will take the mean of
    absolute biases, which measures the magnitude of bias, regardless of direction.
    Finally, the effect size is calculated.
    '''
    bias = np.array(male_preds) - np.array(female_preds)
    total_bias = np.mean(bias)
    absolute_bias = np.mean(np.abs(bias))
    effect_size = np.sum(bias > 0) / len(bias)
    return bias, total_bias, absolute_bias, effect_size


# Main script

def main():

    # import metrics and predictions
    metrics = pd.read_csv("results/model_metrics.csv")
    preds = pd.read_csv("results/model_predictions.csv")

    # Drops row with index 2 and 5, as I forgot to write header=not, when saving results
    metrics.drop([2, 5, 8], inplace = True) 
    preds.drop([2, 5, 8], inplace = True)

    # Convert predictions
    preds['Predictions'] = preds['Predictions'].apply(lambda x: convert_predictions(eval(x)))

    # List all models
    models = ['pretrained', 'original', 'neutral', 'mix']

    # Initialize dict to store results for bias calculations
    bias_results = {'Model': [],
                    'Total_Bias': [],
                    'Abs_Bias': [],
                    'Effect_Size': []}

    # Loop over the models and calculate bias
    for model in models:
        
        # Filter female and male predictions for each model
        female_preds = preds[(preds['Gender'] == 'female') & (preds['Model'] == model)]['Predictions'].values[0]
        male_preds = preds[(preds['Gender'] == 'male') & (preds['Model'] == model)]['Predictions'].values[0]
        
        # Calculate bias and metrics for each model condition
        bias, total_bias, absolute_bias, effect_size = calculate_bias(male_preds, female_preds)

        # Store results
        bias_results['Model'].append(model)
        bias_results['Total_Bias'].append(total_bias)
        bias_results['Abs_Bias'].append(absolute_bias)
        bias_results['Effect_Size'].append(effect_size)

    bias_df = pd.DataFrame(bias_results)
    bias_df.to_csv('results/bias.csv')
    print('Bias is calculated for all models')

if __name__ == "__main__":
    main()
