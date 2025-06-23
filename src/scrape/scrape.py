import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
import re
import argparse

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

def parse_giallozafferano_recipe(url):
    """
    Given the URL of a GialloZafferano recipe page,
    return a dictionary containing:
      - 'presentation'
      - 'ingredients'
      - 'steps'
      - 'featured_data' (parsed from .gz-featured-data-cnt)
    """
    recipe_data = {
        "url": url,
        "presentation": None,
        "ingredients": [],
        "steps": [],
    }

    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # --- 1) PRESENTATION (first <p> in .gz-content-recipe.gz-mBottom4x) ---
    presentation_block = soup.select_one("div.gz-content-recipe.gz-mBottom4x")
    if presentation_block:
        p_tags = presentation_block.find_all("p", recursive=False)
        if p_tags:
            paragraphs = [p.get_text(" ", strip=True) for p in p_tags]
            recipe_data["presentation"] = "\n\n".join(paragraphs)
        else:
            recipe_data["presentation"] = presentation_block.get_text(" ", strip=True)
        
        # suggested_url_block = presentation_block.find("ul")
        suggested_urls_list = []
        suggested_urls = presentation_block.find_all("a")
        for sugg_url_elem in suggested_urls:
            if sugg_url_elem.get('class') is None:
                sugg_url = sugg_url_elem.attrs['href']
                if 'ricette.giallozafferano' in sugg_url:
                    suggested_urls_list.append(sugg_url)
        
        recipe_data['suggested_urls'] = suggested_urls_list

        translation_link = presentation_block.find('a', attrs={'id': 'gz-translation-link'})['href']
        ...

    # --- 2) INGREDIENTS (from .gz-ingredients.gz-mBottom4x.gz-outer) ---
    ingredients_container = soup.select_one("div.gz-ingredients.gz-mBottom4x.gz-outer")
    if ingredients_container:
        ingredient_items = ingredients_container.select("dd.gz-ingredient")
        for dd in ingredient_items:
            txt = dd.get_text(" ", strip=True)
            # Clean excessive spacing
            txt = re.sub(r'\s{2,}', ' ', txt)
            if txt:
                recipe_data["ingredients"].append(txt)

    # --- 3) steps STEPS (in #gz-photocomments-trigger .gz-content-recipe-step) ---
    steps_container = soup.select("div.gz-content-recipe.gz-mBottom4x#gz-photocomments-trigger div.gz-content-recipe-step")
    for step_block in steps_container:
        p_tag = step_block.find("p")
        if p_tag:
            step_text = p_tag.get_text(" ", strip=True)
            recipe_data["steps"].append(step_text)

    # --- 4) FEATURED DATA (.gz-featured-data-cnt) for calories, difficulty, cost, etc. ---
    featured_container = soup.select_one("div.gz-featured-data-cnt")
    if featured_container:
        # Example: parse "calories" from .gz-text-calories-total span
        cal_span = featured_container.select_one(".gz-text-calories-total span")
        if cal_span:
            recipe_data["calories"] = cal_span.get_text(strip=True)

        # Additional info (difficulty, cost, etc.) can appear in .gz-list-featured-data or .gz-list-featured-data-other
        # We gather all <li> texts. Example: "Difficoltà: Facile"
        info_items = featured_container.select(".gz-list-featured-data ul li, .gz-list-featured-data-other ul li")
        for li in info_items:
            text = li.get_text(" ", strip=True)
            # Example: "Difficoltà: Facile" => key="Difficoltà", val="facile"
            if ":" in text:
                key, val = text.split(":", 1)
                if len(key.split()) > 5:
                    # some of the lines have ":" but misplaced
                    # so work around this by filtering out keys that are way too long
                    recipe_data.setdefault("other", []).append(text)
                else:
                    key_stripped = re.sub(' ', '_', key.strip().lower())
                    if key_stripped in key_dict.keys():
                        key_translated = key_dict[key_stripped]
                    else:
                        print(f'metadata key {key_stripped} not in translation dict')
                        key_translated = key_stripped
                    recipe_data[key_translated] = re.sub(' ', '_', val.strip().lower())
            else:
                # Some lines might be stand-alone notes (e.g. "Senza glutine", "Senza lattosio")
                # We'll collect them under a generic 'other' list:
                recipe_data.setdefault("other", []).append(text)

    return recipe_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to scrape recipes from GialloZafferano.it")
    parser.add_argument("--num_recipes", type=int, help="The number of recipes to scrape, used for quick testing", default = 0)
    args = parser.parse_args()

    urls = [el.strip() for el in open('./data/extracted_urls.txt', 'r', encoding='utf8').readlines() if el]
    if args.num_recipes:
        urls = urls[:args.num_recipes]
    all_data = []
    for link in tqdm(urls):
        try:
            data = parse_giallozafferano_recipe(link)
            all_data.append(data)
        except Exception as exc:
            print(f"Error parsing {link}: {exc}")

    with open("./data/gz.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
