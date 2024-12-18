# NLP-exam
...

## 🔧 Setup and installation guide

To ensure reproducibility and facilitate collaboration, follow these steps:

### 1.  Clone the repository
Start by cloning the repository to your local machine using the following command:
```python
$ git clone "https://github.com/SMosegaard/NLP-exam.git"
```
### 2. Set up the virtual environment
Next, execute the setup.sh script to create a virtual environment and install all necessary dependencies listed in requirements.txt:
```python
$ source setup.sh
``` 

## 👩‍💻 Get started
After setting up the project, follow these steps to scrape, clean, and mask the movie review data.

### 1. Scrape movie reviews
To scrape movie reviews from ekkofilm.dk, run the following script:
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
Note: The ```data_masking.py``` script uses ```data_prep/term_lists.py```. To see how the term list is created, see the notebook ```data_prep/data_prep.ipynb```.

### Alternatively! Acces the cleaned, masked data from Hugging Face
If you prefer not to run the scripts above, you can access the cleaned and masked data for training and testing directly from Hugging Face:
```python
from datasets import load_dataset

my_token = {private token provided in the exam paper}
dataset = load_dataset("SMosegaard/ekkofilm-dataset-NLPexam", token = my_token)
``` 

## 🖥️ Model fine-tuning

To fine-tune the model on the different data conditions, use the run.sh script and specify the training data (--data / -d) and whether you want to perform hyperparameter tuning (--hyperparameter_tuning / -ht).

```python
$ source run.sh -d {original_train/neutral_train/mix_train} -ht {yes/no}
```

You can choose from three available datasets: original_train, neutral_train, or mix_train. If you want to perform hyperparameter tuning, please write '-ht yes' and contrary, '-ht no' if not.

The script will automatically convert the input to lowercase, so whether you type the options with capital letters or not, it will not affect the execution.

Based on the user input, the script will perform hyperparameter tuning. Afterwards, the model will be fine-tuned on the best parameters obtained through the tuning or simply use default parameters.

The fine-tuned models will be saved in the folder ```finetuned_models/BERT_finetuned_{data}```.


## 🎬 Model testing: sentiment classification task


After fine-tuning, you can test the model's performance on the sentiment classification task.

The sentiment predictions and test metrics (e.g., accuracy, precision, recall, F1 score) will be saved as pickle files in the folder ```results```.

## ⚥ Bias measure


## Acknowledgments

This repo is a replication of 

Jentzsch, S. F., & Turan, C. (2022). Gender Bias in BERT-Measuring and Analysing Biases through Sentiment Rating in a Realistic Downstream Classification Task. GeBNLP 2022, 184.

