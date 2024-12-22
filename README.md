# 📚 NLP-exam: Assessing Representational Gender Bias in Danish BERT model

This repository contains the code for my exam project in 'Natrual Language Processing'.

## 🔧 Setup and installation guide

To set up the project and ensure reproducibility, follow these steps:

### 1.  Clone the repository
Start by cloning the repository to your local machine using the following command:
```python
$ git clone "https://github.com/SMosegaard/NLP-exam.git"
```
### 2. Set up the virtual environment
Next, execute the ```setup.sh``` script to create a virtual environment and install all necessary dependencies listed in ```requirements.txt```:
```python
$ source setup.sh
``` 
You are now working within the virtual environment.

## 👩‍💻 Get started
After setting up the project, follow these steps to scrape, clean, and mask the movie review data.

### 1. Scrape movie reviews
To scrape movie reviews from ```ekkofilm.dk```, run the following script:
```python
$ python scraping/scrape_reviews.py
```

### 2. Clean the data
Once the data is scraped, use the following script to clean it:
```python
$ python data_prep/data_cleaning.py
``` 

### 3. Create data versions
Next, create different versions of the data (e.g., masked versions) for training and testing by running:
```python
$ python data_prep/data_masking.py
``` 
**Note**: The ```data_masking.py``` script utilizes ```data_prep/term_lists.py```. To see how the term list is created, see the notebook ```data_prep/data_prep.ipynb```.

### Alternatively! Acces the cleaned, masked data from Hugging Face
If you prefer not to run the scripts above, you can access and download the preprocessed datasets directly from Hugging Face:
```python
from datasets import load_dataset

my_token = {private_token_provided_in_the_exam_paper}
dataset = load_dataset("SMosegaard/ekkofilm-dataset-NLPexam", token = my_token)
``` 

## 🖥️ Model fine-tuning

To fine-tune the model on the different data conditions, use the following command and specify the training data (--data / -d) and whether you want to perform hyperparameter tuning (--hyperparameter_tuning / -ht):
```python
$ python model_training/BERT_finetuning.py -d {original/neutral/mix} -ht {yes/no}
```
You can choose from three available datasets: ```original```, ```neutral```, or ```mix```. If you want to perform hyperparameter tuning, please write *'-ht yes'* and contrary, *'-ht no'* if not.

The script will automatically convert the input to lowercase, so whether you type the options with capital letters or not, it will not affect the execution.

Based on the user input, the model will be fine-tuned with the best parameters obtained through the hyperparameter tuning or simply with default parameters. The fine-tuned models will be saved in the folder ```finetuned_models/BERT_finetuned_{data}```.

## 🎬 Model testing: sentiment classification task

Now, you can test the pretrained or fine-tuned models' performance on the sentiment classification task. To do so, you need to specify the model type (--model_type / -mt). If you want to test a fine-tuned model, you will need to further specific which one (--model / -m). 

```python
$ python model_testing/model_testing.py -mt {pretrained/finetuned} -m {original/neutral/mix}
``` 
The sentiment predictions and test metrics (e.g., accuracy, precision, recall, F1 score) will be saved as ```.csv``` files in the folder ```results```.

## ⚥ Bias measure

Finally, bias can be measured for all tested models:
```python
$ python model_testing/bias.py
```
The calculated bias will be saved as a ```.csv``` file in the ```results``` folder.

When finished, deactivate the virtual environment:
```python
$ deactivate
```

## 🎨 Optional: plotting

In the folder ```plotting```, two notebooks designed for visualization tasks can be found: one focuses on plotting the distribution of ratings, while the other visualizes the total and absolute bias across the tested models.

## ⭐ Acknowledgments

This study replicates and extends the work of Jentzsch and Turan (2023) in the context of Danish:

Jentzsch, S. F., & Turan, C. (2022). Gender Bias in BERT-Measuring and Analysing Biases through Sentiment Rating in a Realistic Downstream Classification Task. GeBNLP 2022, 184.