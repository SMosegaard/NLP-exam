import pandas as pd
import numpy as np

def remove_author_prefix(df):
    '''
    Removes the "Af" before the author in the 'Author' column in the df
    '''
    df['Author'] = df['Author'].str.lstrip('Af ')
    return df


def add_sentiment_column(df):
    '''
    Adds a 'Sentiment' column to the df based on the 'Rating' column.
    Ratings 1-3 are considered 'neg' (negative) and ratings 5-6 are considered 'pos' (positive)
    '''
    df['Sentiment'] = np.where(df['Rating'] >= 5, 'pos', np.where(df['Rating'] <= 3, 'neg', 'neu'))
    return df


def add_id_column(df):
    '''
    Adds an 'ID' column to the df based on the index.
    '''
    df['ID'] = df.index
    return df


def date_converter(df):
    '''
    Converts the 'Date' column to a pandas datetime format. Handles Danish month abbreviations 
    and other formatting issues such as removing extra characters and adding spaces
    '''
    month_map = {'jan.': 'Jan', 'feb.': 'Feb', 'mar.': 'Mar', 'apr.': 'Apr', 
                'maj.': 'May', 'jun.': 'Jun', 'jul.': 'Jul', 'aug.': 'Aug', 
                'sep.': 'Sep', 'okt.': 'Oct', 'nov.': 'Nov', 'dec.': 'Dec'}

    df['Date'] = df['Date'].replace(month_map, regex = True)
    df['Date'] = df['Date'].str.replace(r' \|.*', '', regex = True)
    df['Date'] = df['Date'].str.replace(r'(\D)(\d{4})$', r'\1 \2', regex = True)

    df['Date'] = pd.to_datetime(df['Date'], format = '%d. %b %Y')
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

    return df


def clean_text_columns(df):
    '''
    Converts the text to lowercase and removes punctuation and special characters
    '''

    df['Review_cleaned'] = df['Review'].str.lower()

    punc = r'[^\w\s]'
    df['Review_cleaned'] = df['Review_cleaned'].replace(punc, '', regex = True)

    return df


def load_and_process_data(filepath):
    '''
    Loads the data in a .csv format from the specified filepath and applies all transformations
    '''
    df = pd.read_csv(filepath)

    df = remove_author_prefix(df)
    df = add_sentiment_column(df)
    df = add_id_column(df)
    df = date_converter(df)
    df = clean_text_columns(df)

    return df


def main():

    filepath = '/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data/scraped_ekkofilm_reviews.csv'
    #filepath = "data/scraped_ekkofilm_reviews.csv'"
    
    df = load_and_process_data(filepath)

    outpath = "/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data/cleaned_reviews.csv"
    #outpath = "data/reviews.csv"
    df.to_csv(outpath, index = False)
    print("Processed data saved to:", outpath)

if __name__ == "__main__":
    main()
