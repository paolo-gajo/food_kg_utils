import requests
import json
from bs4 import BeautifulSoup
import bs4
import pandas as pd
from tqdm.auto import tqdm
import re
import argparse
import concurrent.futures
import os
from PIL import Image
import numpy as np
from io import BytesIO
from typing import List, Dict
import cv2
from pathlib import Path

class Recipe:
    def __init__(self, id):
        self.presentation_it = None
        self.presentation_it_img_path = ''
        self.ingredients_it = []
        self.steps_it = []
        self.steps_it_img_path = []
        self.presentation_urls_it = set()
        self.related_urls_it = set()
        self.img_count_it = 0
        self.url_en = ''
        self.presentation_en = None
        self.presentation_en_img_path = ''
        self.ingredients_en = []
        self.steps_en = []
        self.steps_en_img_path = []
        self.presentation_urls_en = set()
        self.related_urls_en = set()
        self.img_count_en = 0
        self.other = []
        self.id = id
        self.num_splits = 3

class Scraper:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.key_dict = {
            "difficoltà": "difficulty",
            "preparazione": "prep_time",
            "cottura": "cook_time",
            "dosi_per": "portions",
            "costo": "cost",
            "nota_per_le_calorie": "calories_note",
            "nota_dalla_redazione": "editors_note",
            "nota_note_dalla_redazione": "editors_note",
        }

    def scrape_one(self, url, i):
        # try:
        return self.parse_giallozafferano_recipe(url, i)
        # except Exception as exc:
        #     print(f"Error parsing {url}: {exc}")
        #     return None

    def download_file(self, url):
        response = requests.get(url)
        if response.status_code == 403:
            img_white = np.ones((50, 50, 3), dtype=np.uint8) * 255
            success, encoded_image = cv2.imencode('.png', img_white)
            content = encoded_image.tobytes()
            return content, 1
        else:
            response.raise_for_status()
            return response.content, 0

    def _parse_recipe_page(self, soup, recipe, lang):
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

        pres_img_container = soup.select_one('picture.gz-featured-image img')
        if pres_img_container:
            pres_img_url = pres_img_container.attrs['src']
            content, dl_failed = self.download_file(pres_img_url)
            title = pres_img_url.split('/')[-1]
            title = f'failed_{title}' if dl_failed else title
            filename = os.path.join(str(recipe.id), 'imgs', lang, 'presentation', title)
            filename = self.ensure_extension(filename)
            savename = os.path.join(self.save_dir, filename)
            setattr(recipe, f'presentation_{lang}_img_path', savename)
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            with open(savename, 'wb') as f:
                f.write(content)

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
            step_set = set()
            p_tag_step_list = step_block.find_all("p")
            content_list = []
            flattened_formatted = []
            for p_tag in p_tag_step_list:
                if p_tag:
                    contents = p_tag.contents
                    flattened = self.flatten_contents(contents)
                    for i in range(len(flattened)):
                        if flattened[i].name == 'span' and 'class' in flattened[i].attrs.keys() and 'num-step' in flattened[i].attrs['class']:
                            step_tag = f'<{flattened[i].text}>'
                            flattened_formatted.append(step_tag)
                            step_set.add(step_tag)
                        elif isinstance(flattened[i], bs4.element.Tag):
                            flattened_formatted.append(flattened[i].text)
                        else:
                            flattened_formatted.append(flattened[i])
                    
            if not any(['<' in el for el in flattened_formatted]):
                ...
            content_list += flattened_formatted
            step_text = ' '.join(content_list)
            step_text = re.sub('\s+', ' ', step_text)
            data["steps"].append(step_text)
            img_elem_full = step_block.select_one('div.gz-content-recipe-step-img-container picture.gz-content-recipe-step-img.gz-content-recipe-step-img-full')
            if img_elem_full:
                img_elem = img_elem_full.select_one('img')
                step_img_url_full = img_elem.attrs['src']
                if len(step_set) != recipe.num_splits:
                    print(f"Check recipe {recipe.id}: {getattr(recipe, f'url_{lang}')} (num_splits != 3), set: {step_set}", flush=True)
                self.download_full_step(step_img_url_full, recipe, lang, recipe.num_splits)
            else:
                img_elem_single_list = step_block.select('div.gz-content-recipe-step-img-container picture.gz-content-recipe-step-img.gz-content-recipe-step-img-single')
                for img_elem_single in img_elem_single_list:
                    img_elem = img_elem_single.select_one('img')
                    step_img_url_single = img_elem.attrs['src']
                    self.download_single_steps(step_img_url_single, recipe, lang)

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

    def flatten_contents(self, content_list: List):
        flattened = []
        for cont in content_list:
            children = list(cont.children) if hasattr(cont, 'children') else None
            if not hasattr(cont, 'contents') or not any([hasattr(el, 'contents') for el in children]):
                flattened.append(cont)
            else:
                prev_flat = self.flatten_contents(cont)
                assert isinstance(prev_flat, list)
                flattened += prev_flat
        return flattened 

    def download_single_steps(self, url, recipe, lang):
        if not url.startswith('https'):
            url = 'https://ricette.giallozafferano.it' + url
        content, dl_failed = self.download_file(url)
        
        title = url.split('/')[-1]
        if dl_failed:
            title = f'failed_{title}'
        img_count_field = f'img_count_{lang}'
        img_count = getattr(recipe, img_count_field)
        filename = os.path.join(str(recipe.id), 'imgs', lang, 'steps', f'{img_count}_{title}')    
        filename = self.ensure_extension(filename)
        savename = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        
        with open(savename, 'wb') as f:
            f.write(content)

        setattr(recipe, img_count_field, img_count + 1)
        
        attr_name = f'steps_{lang}_img_path'
        current = getattr(recipe, attr_name, [])
        current += [savename]
        setattr(recipe, attr_name, current)

    def ensure_extension(self, filename: str, extension = 'jpg'):
        if not Path(filename).suffix:
            filename += f'.{extension}'
        return filename

    def download_full_step(self, url, recipe, lang, num_splits):
        if not url.startswith('https'):
            url = 'https://ricette.giallozafferano.it' + url
        
        content, dl_failed = self.download_file(url)
        
        title = url.split('/')[-1]
        title = f'failed_{title}' if dl_failed else title

        img = Image.open(BytesIO(content))
        img_array = np.asarray(img, dtype=np.uint8)
        if len(img_array.shape) < 3:
            img_array = img_array[..., np.newaxis]
        w_split = img_array.shape[1] // num_splits
        img_path_list = []

        img_count_field = f'img_count_{lang}'
        img_count = getattr(recipe, img_count_field)
        counter = 0
        for i in range(num_splits):
            img_array_split = img_array[:,i*w_split:(i+1)*w_split:,:]
            # title = re.sub(img_range, str(i), title)
            filename = os.path.join(str(recipe.id), 'imgs', lang, 'steps', f'{img_count + counter}_{title}')    
            filename = self.ensure_extension(filename)
            savename = os.path.join(self.save_dir, filename)
            img_path_list.append(savename)
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            Image.fromarray(img_array_split).save(savename)
            counter += 1
        setattr(recipe, img_count_field, img_count + counter)
        attr_name = f'steps_{lang}_img_path'
        current = getattr(recipe, attr_name, [])
        current += img_path_list
        setattr(recipe, attr_name, current)

    def parse_giallozafferano_recipe(self, url, i):
        recipe = Recipe(id=i)

        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        recipe.url_it = url
        recipe.title_it = recipe.url_it[recipe.url_it.rfind('/') + 1:recipe.url_it.rfind('.')].replace('-', ' ')
        it_data = self._parse_recipe_page(soup, recipe=recipe, lang='it')
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
                        key_translated = self.key_dict.get(key_stripped, key_stripped)
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
            recipe.url_en = url_en
            recipe.title_en = recipe.url_en[recipe.url_en.rfind('/') + 1:recipe.url_en.rfind('.')].replace('-', ' ')
            en_data = self._parse_recipe_page(soup_translated, recipe=recipe, lang='en')
            recipe.presentation_en = en_data["presentation"]
            recipe.ingredients_en = en_data["ingredients"]
            recipe.steps_en = en_data["steps"]
            recipe.presentation_urls_en = recipe.presentation_urls_en | en_data["presentation_urls"]
            recipe.related_urls_en = recipe.related_urls_en | en_data["related_urls"]
        
        self.save_recipe(self.make_recipe_dict(recipe))
        return recipe

    def make_recipe_dict(self, recipe):
        serializable_dict = {}
        for k, v in recipe.__dict__.items():
            if isinstance(v, set):
                serializable_dict[k] = list(v)
            else:
                serializable_dict[k] = v
        return serializable_dict

    def save_recipe(self, recipe: Dict):
        filename = os.path.join(str(recipe['id']), 'recipe', f"{str(recipe['id'])}_{recipe['title_it']}.json")
        savename = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        with open(savename, 'w', encoding='utf8') as f:
            json.dump(recipe, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to scrape recipes from GialloZafferano.it")
    parser.add_argument("--num_recipes", type=int, help="The number of recipes to scrape, used for quick testing", default=0)
    parser.add_argument("--num_workers", type=int, help="Number of parallel threads to use", default=4)
    args = parser.parse_args()

    urls = [el.strip() for el in open('./misc/gz_urls.txt', 'r', encoding='utf8').readlines() if el]
    if args.num_recipes:
        urls = urls[:args.num_recipes]
    
    save_dir = './data/gz_dataset'
    scraper = Scraper(save_dir=save_dir)

    # all_data = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    #     futures = list(tqdm(executor.map(scrape_one, urls), total=len(urls)))
    #     all_data = [dict(res) for res in futures if res is not None]
    all_data = []
    for i, url in tqdm(enumerate(urls), total=len(urls)):
        scraped_recipe = scraper.scrape_one(url, i)
        scraped_recipe_dict = scraper.make_recipe_dict(scraped_recipe)
        if scraped_recipe_dict:
            all_data.append(scraped_recipe_dict)

    with open("./data/gz_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
