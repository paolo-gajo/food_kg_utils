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
    try:
        return parse_giallozafferano_recipe(url)
    except Exception as exc:
        print(f"Error parsing {url}: {exc}")
        return None

def _parse_recipe_page(soup):
    """Parses the core components of a GZ recipe soup into a partial dictionary."""
    data = {
        "presentation": None,
        "ingredients": [],
        "steps": [],
        "suggested_urls": [],
    }

    # --- 1) Presentation ---
    presentation_block = soup.select_one("div.gz-content-recipe.gz-mBottom4x")
    if presentation_block:
        p_tags = presentation_block.find_all("p", recursive=False)
        if p_tags:
            paragraphs = [p.get_text(" ", strip=True) for p in p_tags]
            data["presentation"] = "\n\n".join(paragraphs)
        else:
            data["presentation"] = presentation_block.get_text(" ", strip=True)

        suggested_urls = presentation_block.find_all("a")
        for sugg_url_elem in suggested_urls:
            if sugg_url_elem.get('class') is None:
                if 'href' in sugg_url_elem.attrs.keys():
                    sugg_url = sugg_url_elem.attrs['href']
                    if 'ricette.giallozafferano' in sugg_url:
                        data["suggested_urls"].append(sugg_url)

    # --- 2) Ingredients ---
    ingredients_container = soup.select_one("div.gz-ingredients.gz-mBottom4x.gz-outer")
    if ingredients_container:
        ingredient_items = ingredients_container.select("dd.gz-ingredient")
        for dd in ingredient_items:
            txt = dd.get_text(" ", strip=True)
            txt = re.sub(r'\s{2,}', ' ', txt)
            if txt:
                data["ingredients"].append(txt)

    # --- 3) Steps ---
    steps_container = soup.select("div.gz-content-recipe.gz-mBottom4x div.gz-content-recipe-step")
    for step_block in steps_container:
        p_tag = step_block.find("p")
        if p_tag:
            step_text = p_tag.get_text(" ", strip=True)
            data["steps"].append(step_text)

    return data

def parse_giallozafferano_recipe(url):
    recipe_data = {
        "url_it": url,
        "presentation_it": None,
        "ingredients_it": [],
        "steps_it": [],
        "suggested_urls_it": [],
        "url_en": '',
        "presentation_en": None,
        "ingredients_en": [],
        "steps_en": [],
        "suggested_urls_en": [],
    }

    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    it_data = _parse_recipe_page(soup)
    recipe_data["presentation_it"] = it_data["presentation"]
    recipe_data["ingredients_it"] = it_data["ingredients"]
    recipe_data["steps_it"] = it_data["steps"]
    recipe_data["suggested_urls_it"] = it_data["suggested_urls"]

    # --- Featured data (unique to IT page) ---
    featured_container = soup.select_one("div.gz-featured-data-cnt")
    if featured_container:
        cal_span = featured_container.select_one(".gz-text-calories-total span")
        if cal_span:
            recipe_data["calories"] = cal_span.get_text(strip=True)

        info_items = featured_container.select(".gz-list-featured-data ul li, .gz-list-featured-data-other ul li")
        for li in info_items:
            text = li.get_text(" ", strip=True)
            if ":" in text:
                key, val = text.split(":", 1)
                if len(key.split()) > 5:
                    recipe_data.setdefault("other", []).append(text)
                else:
                    key_stripped = re.sub(' ', '_', key.strip().lower())
                    key_translated = key_dict.get(key_stripped, key_stripped)
                    recipe_data[key_translated] = re.sub(' ', '_', val.strip().lower())
            else:
                recipe_data.setdefault("other", []).append(text)

    # --- Translate page ---
    presentation_block_en = soup.select_one("div.gz-content-recipe.gz-mBottom4x")
    translation_link = presentation_block_en.find('a', attrs={'id': 'gz-translation-link'})
    if translation_link and translation_link.attrs['href']:
        url_en = translation_link.attrs['href']
        resp = requests.get(url_en)
        resp.raise_for_status()
        soup_translated = BeautifulSoup(resp.text, "html.parser")
        en_data = _parse_recipe_page(soup_translated)
        recipe_data["url_en"] = url_en
        recipe_data["presentation_en"] = en_data["presentation"]
        recipe_data["ingredients_en"] = en_data["ingredients"]
        recipe_data["steps_en"] = en_data["steps"]
        recipe_data["suggested_urls_en"] = en_data["suggested_urls"]
    return recipe_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to scrape recipes from GialloZafferano.it")
    parser.add_argument("--num_recipes", type=int, help="The number of recipes to scrape, used for quick testing", default=0)
    parser.add_argument("--num_workers", type=int, help="Number of parallel threads to use", default=4)
    args = parser.parse_args()

    urls = [el.strip() for el in open('./data/extracted_urls.txt', 'r', encoding='utf8').readlines() if el]
    if args.num_recipes:
        urls = urls[:args.num_recipes]

    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = list(tqdm(executor.map(scrape_one, urls), total=len(urls)))
        all_data = [res for res in futures if res is not None]
    # all_data = []
    # for url in tqdm(urls):
    #     out = scrape_one(url)
    #     if out:
    #         all_data.append(out)

    with open("./data/gz_bilingual.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
