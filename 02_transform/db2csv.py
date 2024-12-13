import os
os.chdir("./02_transform")

import sqlite3
import pandas as pd
from tqdm import tqdm

dbnames = ["Jan","janin","jay","jiwoo","overall"]
conns = []

cache = {}



for dbname in dbnames:
    conn = sqlite3.connect(f"../01_inference/groq/{dbname}.db")
    conns.append(conn)
    c = conn.cursor()
    c.execute('''
        SELECT model, prompt, response FROM cache
    ''')
    res = c.fetchall()
    for r in res:
        cache[(r[0], r[1])] = r[2]

def get_cache(model, prompt):
    for conn in conns:
        c = conn.cursor()
        c.execute('''
            SELECT response FROM cache WHERE model = ? AND prompt = ?
        ''', (model, prompt))
        res = c.fetchone()
        if res:
            return res[0]
    return None

get_cache2_cache = {}
def get_cache2(model, prompt):
    if (model, prompt) in cache:
        return cache[(model, prompt)]
    return None

df = pd.read_parquet("hf://datasets/potsawee/wiki_bio_gpt3_hallucination/data/evaluation-00000-of-00001-e91191b8ff41afbe.parquet")
df_ground = pd.read_csv("with-gpt3-gpt4.csv")

models = ["llama-3.2-3b-preview", "gemma2-9b-it", "mixtral-8x7b-32768", "llama-3.2-3b-preview-rabbit", "llama-3.2-3b-preview-rabbitX"]
PROMPT_TEMPLATE1 = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
PROMPT_TEMPLATE2 = "If you answer this properly, a cute rabbit will be happy 😊🐇✨.If you do it wrong, a cute rabbit will burst into tears 😭🐇💧.\n\nContext: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
PROMPT_TEMPLATE3 = "If you answer this properly, a cute rabbit will be happy. If you do it wrong, a cute rabbit will burst into tears.\n\nContext: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No. Think twice before answering.\n\nAnswer: "

models_prompt_template = {
    "llama-3.2-3b-preview": PROMPT_TEMPLATE1,
    "gemma2-9b-it": PROMPT_TEMPLATE1,
    "mixtral-8x7b-32768": PROMPT_TEMPLATE1,
    "llama-3.2-3b-preview-rabbit": PROMPT_TEMPLATE2,
    "llama-3.2-3b-preview-rabbitX": PROMPT_TEMPLATE3
}
for model in models:
    model_scores = []
    for row in tqdm(df.itertuples(index=True)):
        scores = []
        for s_id, sentence in enumerate(row.gpt3_sentences):
            values = []
            for c_id, context in enumerate(row.gpt3_text_samples):
                context = context.replace("\n", " ") 
                prompt_template = models_prompt_template[model]
                prompt = prompt_template.format(context=context, sentence=sentence)
                cached = get_cache2(model, prompt)
                if not cached:
                    print(row.Index, s_id, c_id, model)
                cached = cached.lower()
                if 'yes' in cached:
                    v = 0
                elif 'no' in cached:
                    v = 1
                else:
                    v = 0.5
                values.append(v)
            score = sum(values) / len(values)
            scores.append(score)
        model_scores.append(scores)
    df_ground[model] = model_scores

df_ground.to_csv("groq-hallucination.csv", index=False)
        

