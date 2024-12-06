import os
from itertools import chain
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import pickle
import nltk
from term_lists import *


def count_terms(review, terms = all_terms):
    '''
    Function that counts occurrences of specified terms in a given review. First, a dictionary
    is initialized with each term set to an initial count of zero. Then, it iterates trough
    the list of terms, checks for occurrences of each term within the review, and updates the
    count accordingly.
    
    The variable 'pattern' ensures, that all terms are counted also if they are embedded in compound
    words ("pige" in "blomsterpigen"). So the term can be a part of a word or standalone.
    '''
    term_counted = dict.fromkeys(terms, 0)
    for term in terms:
        #term = re.escape(term) # to avoid regex issues
        pattern = rf'{term}\w*'
        term_counted[term] = len(re.findall(pattern, review))
    return term_counted


def mask_by_term_dict(review, term_dict):
    '''
    Replace gender-specific terms in the review based on the term_dict
    '''
    for term, replacement in term_dict.items():
        #term = re.escape(term)
        pattern = rf'({term})(\w*)'
        review = re.sub(pattern, replacement + '\\2', review)
    return review


def make_male(review):
    '''
    Mask all female-specific names and terms in the text.
    '''
    return mask_by_term_dict(review, terms_f2m)


def make_female(review):
    '''
    Mask all male-specific names and terms in the text.
    '''
    return mask_by_term_dict(review, terms_m2f)


def make_neutral(review, terms = all_terms):
    '''
    Mask all female- and male-specific names and terms in the text to make it neutral.
    '''
    for term in terms:
        #term = re.escape(term)
        pattern = rf'{term}\w*'
        review = re.sub(pattern, '', review)
    return review


def make_all_df(df, output_path):

    reviews_list = df['Review_cleaned'].tolist()

    # Count occurrences of gendered terms
    df["count_table"] = [count_terms(review, all_terms) for review in reviews_list]
    df["count_total"] = [sum(counts.values()) for counts in df["count_table"].tolist()]
    
    # Count occurrences of pronouns
    df["count_prons"] = [sum(counts[pronoun] for pronoun in all_prons if pronoun in counts) for counts in df["count_table"].tolist()]

    # Mask by dictionary for different gender categories
    df["text_all_male"] = [make_male(review) for review in reviews_list]
    df["text_all_female"] = [make_female(review) for review in reviews_list]
    df["text_all_neutral"] = [make_neutral(review, all_terms) for review in reviews_list]

    masked_df_file_path = os.path.join(output_path, "reviews_masked.csv")
    df.to_csv(masked_df_file_path, index = False)

    return df


def split_and_save(df, output_path):

    train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 123)

    original_train = train_df[['ID', 'Review_cleaned', 'Sentiment']].rename(columns = {'Review_cleaned': 'text'})

    neutral_train = train_df[['ID', 'text_all_neutral', 'Sentiment']].rename(columns = {'text_all_neutral': 'text'})

    mix_train = pd.concat([
        train_df[['ID', 'text_all_male', 'Sentiment']].rename(columns = {'text_all_male': 'text'}),
        train_df[['ID', 'text_all_female', 'Sentiment']].rename(columns = {'text_all_female': 'text'})])

    all_male_test = test_df[['ID', 'text_all_male', 'Sentiment']].rename(columns = {'text_all_male': 'text'})
    all_female_test = test_df[['ID', 'text_all_female', 'Sentiment']].rename(columns = {'text_all_female': 'text'})

    original_train.to_csv(os.path.join(output_path, "original_train.csv"), index = False)
    neutral_train.to_csv(os.path.join(output_path, "neutral_train.csv"), index = False)
    mix_train.to_csv(os.path.join(output_path, "mix_train.csv"), index = False)
    all_male_test.to_csv(os.path.join(output_path, "all_male_test.csv"), index = False)
    all_female_test.to_csv(os.path.join(output_path, "all_female_test.csv"), index = False)


def main():

    filepath = "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/cleaned_reviews.csv"
    #filepath = "data/reviews.csv'"

    df = pd.read_csv(filepath)
    df = df[df.Sentiment != 'neu'] # only interested reviews with pos and neg sentiment

    output_path = "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data_2/"
    #outpath = "data/reviews_masked.csv"
    masked_df = make_all_df(df, output_path)

    split_and_save(masked_df, output_path)

    print("Data processing and saving complete")

if __name__ == "__main__":
    main()