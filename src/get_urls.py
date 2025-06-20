import requests
from bs4 import BeautifulSoup
import time
from tqdm.auto import tqdm

def extract_urls(base_url, max_pages, min_pages = 1):
    urls = []
    pbar = tqdm(range(min_pages, max_pages + 1))
    for page in pbar:
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}page{page}/"
        
        # print(f"Scraping: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        recipe_links = soup.select("h2.gz-title a")

        if len(recipe_links) == 0:
            break
        
        for link in recipe_links:
            urls.append(link['href'])
        
        time.sleep(1)  # To avoid overloading the server
        pbar.set_description(f'Scraping URL: {url}')
    return urls

if __name__ == "__main__":
    BASE_URL = "https://www.giallozafferano.it/ricette-cat/"
    MIN_PAGES = 1  # Last page
    MAX_PAGES = 10000  # Last page
    
    extracted_urls = extract_urls(BASE_URL, MAX_PAGES, MIN_PAGES)
    
    # Save to file
    with open("./data/extracted_urls.txt", "w") as f:
        for url in extracted_urls:
            f.write(url + "\n")
    
    print(f"Extracted {len(extracted_urls)} URLs. Saved to extracted_urls.txt")
