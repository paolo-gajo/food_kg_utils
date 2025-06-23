# pip install wikipedia
# python -m spacy download it_core_news_md

import pandas as pd
import spacy
import wikipedia 

nlp = spacy.load("it_core_news_md")
# ner = nlp.add_pipe("ner")

df = pd.read_json("data/gz_locs.json")

def url_to_query(url: str):
    name = url[url.rfind("/") + 1:url.rfind(".")]
    return name.replace("-", " ")

wikipedia.set_lang("it")

go = False
for i, recipe in df.iterrows():
    # print(recipe.url)
    if recipe.url == "https://ricette.giallozafferano.it/Pizzette-di-melanzane.html":
        go = True 
    if go:
        if recipe.country == "Italia" and recipe.region == "UNK":
            wikiquery = url_to_query(recipe["url"])
            print(recipe["url"], wikiquery)
            results = wikipedia.search(wikiquery)
            for i, article in enumerate(results):
                print(i, article)

            # ask for candidate to display(
            option = int(input("article to load:"))
            if option == -1:
                continue
            summ = wikipedia.summary(results[option])
            print(summ)
            processed = nlp(summ)
            # print(processed)
            for ent in processed.ents:
                print(ent.text, ent.start_char, ent.end_char, ent.label_)
            print("\n", results[option], "\n")

            print("\n\n\n")
            # if "exit" -> interrupt for and save the file
            # query the summary of the selected candidate 
            # input the region (if empty, keep UNK)
            # exit()
# print(df)