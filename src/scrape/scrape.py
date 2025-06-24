import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
import re
import argparse
import concurrent.futures

key_dict = {
    "difficoltà": "difficulty",
    "preparazione": "prep_time",
    "cottura": "cook_time",
    "dosi_per": "portions",
    "costo": "cost",
    "nota_per_le_calorie": "calories_note",
    "nota_dalla_redazione": "editors_note",
    "nota_note_dalla_redazione": "editors_note",
}

def scrape_one(url):
    # try:
    return parse_giallozafferano_recipe(url)
    # except Exception as exc:
    #     print(f"Error parsing {url}: {exc}")
    #     return None

class Recipe:
    def __init__(self, url):
        self.url_it = url
        self.presentation_it = None
        self.ingredients_it = []
        self.steps_it = []
        self.presentation_urls_it = set()
        self.related_urls_it = set()
        self.url_en = ''
        self.presentation_en = None
        self.ingredients_en = []
        self.steps_en = []
        self.presentation_urls_en = set()
        self.related_urls_en = set()
        self.other = []

def _parse_recipe_page(soup):
    """Parses the core components of a GZ recipe soup into a partial dictionary."""
    data = {
        "presentation": None,
        "ingredients": [],
        "steps": [],
        "presentation_urls": set(),
        "related_urls": set(),
    }

    # presentation
    presentation_block = soup.select_one("div.gz-content-recipe.gz-mBottom4x")
    if presentation_block:
        p_tags = presentation_block.find_all("p", recursive=False)
        if p_tags:
            paragraphs = [p.get_text(" ", strip=True) for p in p_tags]
            data["presentation"] = "\n\n".join(paragraphs)
        else:
            data["presentation"] = presentation_block.get_text(" ", strip=True)

        presentation_urls = presentation_block.find_all("a")
        for sugg_url_elem in presentation_urls:
            if sugg_url_elem.get('class') is None:
                if 'href' in sugg_url_elem.attrs.keys():
                    pres_url = sugg_url_elem.attrs['href']
                    if 'ricette.giallozafferano' in pres_url:
                        data["presentation_urls"].add(pres_url)

    # ingredients
    ingredients_container = soup.select_one("div.gz-ingredients.gz-mBottom4x.gz-outer")
    if ingredients_container:
        ingredient_items = ingredients_container.select("dd.gz-ingredient")
        for dd in ingredient_items:
            txt = dd.get_text(" ", strip=True)
            txt = re.sub(r'\s{2,}', ' ', txt)
            if txt:
                data["ingredients"].append(txt)

    # steps
    steps_container = soup.select("div.gz-content-recipe.gz-mBottom4x div.gz-content-recipe-step")
    for step_block in steps_container:
        p_tag = step_block.find("p")
        if p_tag:
            step_text = p_tag.get_text(" ", strip=True)
            data["steps"].append(step_text)
    
    # related urls
    related_container_1 = soup.select("div.gz-swiper-element-shadowed.gz-mBottom3x div.gz-related-swiper")
    for related_block in related_container_1:
        if related_block.attrs['data-swipername'] == 'gz-related-swiper':
            rel_url_elem_list = related_block.find_all('a')
            for rel_url_elem in rel_url_elem_list:
                if rel_url_elem and rel_url_elem.attrs['href']:
                        rel_url_string = rel_url_elem.attrs['href']
                        data['related_urls'].add(rel_url_string)

    related_container_2 = soup.select("div.gz-content.gz-elevator-ame-base section.gz-related.gz-pTop3x")
    for related_block in related_container_2:
        if related_block.attrs['data-swipername'] == 'gz-related':
            rel_url_block_list = related_block.select("h2.gz-title")
            for rel_block_elem in rel_url_block_list:
                rel_url = rel_block_elem.find('a')
                if rel_url and rel_url.attrs['href']:
                    rel_url_string = rel_url.attrs['href']
                    data['related_urls'].add(rel_url_string)

    return data

def parse_giallozafferano_recipe(url):
    recipe = Recipe(url)

    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    it_data = _parse_recipe_page(soup)
    recipe.presentation_it = it_data["presentation"]
    recipe.ingredients_it = it_data["ingredients"]
    recipe.steps_it = it_data["steps"]
    recipe.presentation_urls_it = recipe.presentation_urls_it | it_data["presentation_urls"]
    recipe.related_urls_it = recipe.related_urls_it | it_data["related_urls"]

    # Featured data (unique to IT page) ---
    featured_container = soup.select_one("div.gz-featured-data-cnt")
    if featured_container:
        cal_span = featured_container.select_one(".gz-text-calories-total span")
        if cal_span:
            recipe.calories = cal_span.get_text(strip=True)

        info_items = featured_container.select(".gz-list-featured-data ul li, .gz-list-featured-data-other ul li")
        for li in info_items:
            text = li.get_text(" ", strip=True)
            if ":" in text:
                key, val = text.split(":", 1)
                if len(key.split()) > 5:
                    recipe.other.append(text)
                else:
                    key_stripped = re.sub(' ', '_', key.strip().lower())
                    key_translated = key_dict.get(key_stripped, key_stripped)
                    setattr(recipe, key_translated, re.sub(' ', '_', val.strip().lower()))
            else:
                recipe.other.append(text)

    # Translate page ---
    presentation_block_en = soup.select_one("div.gz-content-recipe.gz-mBottom4x")
    translation_link = presentation_block_en.find('a', attrs={'id': 'gz-translation-link'})
    if translation_link and translation_link.attrs['href']:
        url_en = translation_link.attrs['href']
        resp = requests.get(url_en)
        resp.raise_for_status()
        soup_translated = BeautifulSoup(resp.text, "html.parser")
        en_data = _parse_recipe_page(soup_translated)
        recipe.url_en = url_en
        recipe.presentation_en = en_data["presentation"]
        recipe.ingredients_en = en_data["ingredients"]
        recipe.steps_en = en_data["steps"]
        recipe.presentation_urls_en = recipe.presentation_urls_en | en_data["presentation_urls"]
        recipe.related_urls_en = recipe.related_urls_en | en_data["related_urls"]
    return recipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to scrape recipes from GialloZafferano.it")
    parser.add_argument("--num_recipes", type=int, help="The number of recipes to scrape, used for quick testing", default=0)
    parser.add_argument("--num_workers", type=int, help="Number of parallel threads to use", default=4)
    args = parser.parse_args()

    urls = [el.strip() for el in open('./misc/gz_urls.txt', 'r', encoding='utf8').readlines() if el]
    if args.num_recipes:
        urls = urls[:args.num_recipes]

    all_data = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    #     futures = list(tqdm(executor.map(scrape_one, urls), total=len(urls)))
    #     all_data = [dict(res) for res in futures if res is not None]
    all_data = []
    for url in tqdm(urls):
        out = scrape_one(url)
        out = out.__dict__
        if out:
            all_data.append(out)

    with open("./data/gz_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
