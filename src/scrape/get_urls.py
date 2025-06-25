import requests
from bs4 import BeautifulSoup
import time
from tqdm.auto import tqdm
import json
import os

region_list = [
    "Abruzzo",
    "Basilicata",
    "Calabria",
    "Campania",
    "Emilia-Romagna",
    "Friuli-Venezia Giulia",
    "Lazio",
    "Liguria",
    "Lombardia",
    "Marche",
    "Molise",
    "Piemonte",
    "Puglia",
    "Sardegna",
    "Sicilia",
    "Toscana",
    "Trentino-Alto Adige",
    "Umbria",
    "Valle d'Aosta",
    "Veneto"
    ]

def extract_urls(base_url, min_pages = 1, region = ''):
    urls = []
    url_layout = base_url + r"page{page}/"
    if region:
        url_layout += r'regionali/{region}'
    url_first = url_layout.format(page=str(1), region=region)
    resp = requests.get(url_first)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    page_list_block = soup.select_one('div.gz-nums')
    if page_list_block:
        max_pages_block = page_list_block.select_one('span.disabled.total-pages')
        if max_pages_block:
            max_pages = int(max_pages_block.text)
        else:
            page_list = page_list_block.select('div.gz-pages a.page')
            max_pages = int(page_list[-1].text)
    else:
        max_pages = 1
    pbar = tqdm(range(min_pages, max_pages + 1))
    for page in pbar:
        url = url_layout.format(page=page, region=region)
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
    base_url = "https://www.giallozafferano.it/ricette-cat/"
    new_list = 0
    misc_dir = "./misc"
    gz_urls_path = os.path.join(misc_dir, 'gz_urls.txt')
    if new_list:        
        extracted_urls = extract_urls(base_url, min_pages=1)
        # Save to file
        with open(gz_urls_path, "w") as f:
            for url in extracted_urls:
                f.write(url + "\n")
    else:
        extracted_urls = [el.strip() for el in open(gz_urls_path, 'r').readlines()]

    print(f"Extracted {len(extracted_urls)} URLs. Saved to extracted_urls.txt")

    url_dict = {}
    
    for entry in region_list:
        rgn = entry.replace(' ', '-').replace("'", "-")
        extracted_urls = extract_urls(base_url, min_pages=1, region=rgn)
        url_dict[rgn] = extracted_urls
    
    with open(os.path.join(misc_dir, 'gz_regional_urls.json'), 'w', encoding='utf8') as f:
        json.dump(url_dict, f, ensure_ascii = False, indent = 4)
