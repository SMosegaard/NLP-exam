import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_reviews(urls):
    '''
    Scrape all movie reviews from ekkofilm.dk from a list of URLs
    Input: URL of the review page to scrape.
    Output: 
    First, initialized lists to store the scraped data
    '''
    all_reviews = []

    for url in urls:

        response = requests.get(url)
        response.raise_for_status()  

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title, data, and author by class name
        title = soup.find('h1').get_text(strip = True)
        date = soup.find('div', class_ = 'published').get_text(strip = True)
        author = soup.find('div', class_ = 'author').get_text(strip = True)
        
        # Extract rating by counting 'star-black' images
        rating_div = soup.find('div', class_ = 'rating')
        rating = len(rating_div.find_all('img', src = lambda src: src and 'star-black.png' in src))

        # Extract full review text
        text_div = soup.find('div', class_ = 'text')
        if text_div:
            paragraphs = text_div.find_all('p')
            text = ' '.join([p.get_text(strip = True) for p in paragraphs])
        else:
            text = None

        # Append the extracted data as a tuple
        all_reviews.append((title, url, author, date, rating, text))

    # Create DataFrame from the collected data and specify column names
    df = pd.DataFrame(all_reviews, columns = ['Title', 'URL', 'Author', 'Date', 'Rating', 'Review'])
    
    return df



def main():

    scraped_urls_path = '/work/SofieNørboMosegaard#5741/NLP/NLP-exam/scraping/ekkofilm_urls_webscraper.csv'
    scraped_urls = pd.read_csv(scraped_urls_path, delimiter = ',', header = None)
    
    scraped_urls.columns = ['Title', 'URL']
    urls = scraped_urls['URL'].tolist()

    df = scrape_reviews(urls)

    output_path = '/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data/scraped_ekkofilm_reviews.csv'
    df.to_csv(output_path, index = False)

    print(f"Data scraped and saved successfully")

if __name__ == "__main__":
    main()