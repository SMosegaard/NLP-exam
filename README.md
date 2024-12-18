# NLP-exam
...

## 🔧 Setup and Installation Guide

To ensure reproducibility and facilitate collaboration, follow these steps:

### 1.  Clone the repository
Start by cloning the repository to your local machine using the following command:
```python
$ git clone "https://github.com/SMosegaard/NLP-exam.git"
```
### 2. Set Up the Virtual Environment
Next, execute the setup.sh script to create a virtual environment and install all necessary dependencies listed in requirements.txt:
```python
$ source setup.sh
``` 

## 👩‍💻 Get Started
After setting up the project, follow these steps to scrape, clean, and mask the movie review data.

### 1. Scrape Movie Reviews
To scrape movie reviews from ekkofilm.dk, run the following script:
```python
$ python scraping/scrape_reviews.py
```

### 2. Clean the Data
Once the data is scraped, use the following script to clean it:
```python
$ python data_prep/data_cleaning.py
``` 

### 3. Create Data Versions
Next, create different versions of the data (e.g., masked versions) by running:
```python
$ python data_prep/data_masking.py
``` 
Note: The ```data_masking.py``` script uses ```data_prep/term_lists.py```. To see how the term list is created, see the notebook ```data_prep/data_prep.ipynb```.

### Alternative Access to Cleaned, Masked Data
If you prefer not to run the scripts above, you can access the cleaned and masked data for training and testing directly from Hugging Face:
```python
from datasets import load_dataset

my_token = {private token provided in the exam paper}
dataset = load_dataset("SMosegaard/ekkofilm-dataset-NLPexam", token = my_token)
``` 