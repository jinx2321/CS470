# ==================== INSTRUCTIONS =====================
# 1. Put your GROQ API key in the GROQ_API_KEY variable.
# 2. Uncomment your row range.
# 3. python -m venv .venv
# 4. source .venv/bin/activate
# 5. pip install httpx pandas aiosqlite tqdm pyarrow fastparquet huggingface_hub
# 6. python process_groq.py
# 7. Deactivate the environment by typing "deactivate"
# 8. Rename the "cache.db" into "your_name.db", then commit and push.

# ===================== SETTINGS ========================

GROQ_API_KEY = 'gsk_PutYourGROQAPIkeyherePutYourGROQAPIkeyherePutYourGRO' # Put your GROQ API key here!!

# Uncomment your row range!!

# Jan
# START_ROW, END_ROW = 0, 48

# Jay
# START_ROW, END_ROW = 48, 96

# Janin
# START_ROW, END_ROW = 96, 144

# Seunghwan
# START_ROW, END_ROW = 144, 192

# Jiwoo
# START_ROW, END_ROW = 192, 238

LLM_RESPONSE_MAPPING = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
PROMPT_TEMPLATE = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "

# ====================== CODE ==========================

import httpx
import asyncio
import pandas as pd
import aiosqlite
import time
import re
import zlib
import tqdm.asyncio

df = pd.read_parquet("hf://datasets/potsawee/wiki_bio_gpt3_hallucination/data/evaluation-00000-of-00001-e91191b8ff41afbe.parquet")

async def groq_completion(model, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": model,
        "temperature": 0,
        "max_tokens": 5,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Error in groq_completion: {response.json()}")
    raise Exception("Unreachable: end of function 'groq_completion'")

async def init_db():
    conn = await aiosqlite.connect('cache.db')
    c = await conn.cursor()
    await c.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            prompt TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    await conn.commit()
    return conn

async def insert_cache(conn, model, prompt, response):
    c = await conn.cursor()
    await c.execute('''
        INSERT INTO cache (model, prompt, response) VALUES (?, ?, ?)
    ''', (model, prompt, response))
    await conn.commit()

async def get_cache(conn, model, prompt):
    c = await conn.cursor()
    await c.execute('''
        SELECT response FROM cache WHERE model = ? AND prompt = ?
    ''', (model, prompt))
    return await c.fetchone()

def CRC32(s):
    return f"{zlib.crc32(s.encode()):08X}"

semaphore = asyncio.Semaphore(10)
async def process_prompt(conn, model, prompt):
    async with semaphore:
        cached = await get_cache(conn, model, prompt)
        if cached:
            return cached
        ret = None
        count = 10
        while count >= 0 and ret == None:
            try:
                ret = await groq_completion(model,prompt)
                await insert_cache(conn, model, prompt, ret)
            except Exception as e:
                if match := re.search(r"try again in ([\d\.]+)s", str(e)):
                    wait_time = float(match.group(1))
                    # print(f"Exceed Tokens per Minute limit -> {model:20s} {CRC32(prompt)}: I will wait {0.5 + wait_time:04f} seconds and try again")
                    await asyncio.sleep(0.5 + wait_time)
                elif match := re.search(r"try again in ([\d\.]+)ms", str(e)):
                    wait_time = float(match.group(1)) / 1000
                    # print(f"Exceed Tokens per Minute limit -> {model:20s} {CRC32(prompt)}: I will wait {0.5 + wait_time:04f} ms and try again")
                    await asyncio.sleep(0.5 + wait_time)
                else:
                    raise e

            count -= 1
        if ret == None:
            print(f"DEBUG: Error in process_prompt: I tried 10 times, but still failed, please try again later")
            print(f"DEBUG: Model: {model}")
            print(f"DEBUG: Prompt: {prompt}")
            raise Exception(f"Error in process_prompt: I tried 10 times, but still failed, please try again later")
        return ret
        

async def process_sentence(conn, pbar, model, sentence, contexts):
    ret = []
    for context in contexts:
        context = context.replace("\n", " ") 
        prompt = PROMPT_TEMPLATE.format(context=context, sentence=sentence)
        ret.append(await process_prompt(conn, model, prompt))
        pbar.update(1)
    return ret

async def process_row(conn, model, row):
    ret = []
    with tqdm.asyncio.tqdm(total=len(row.gpt3_sentences)*len(row.gpt3_text_samples), desc=f"{model:20s} {row.Index+1}th row", leave=False, maxinterval=1, unit="inferences", bar_format="{l_bar}{bar} | {n_fmt: >3}/{total_fmt: >3} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for sentence in row.gpt3_sentences:
            ret.append(await process_sentence(conn, pbar, model, sentence, row.gpt3_text_samples))
    return ret

async def process_rows(conn, model, s,e):
    ret = []
    for row in df.itertuples(index=True):
        if row.Index < s:
            continue
        if row.Index >= e:
            break
        ret.append(await process_row(conn, model, row))
    return ret

async def main():
    conn = await init_db()
    try:
        models = ["llama-3.2-1b-preview", "gemma2-9b-it"]

        await asyncio.gather(
            *(process_rows(conn, model, START_ROW, END_ROW) for model in models)
        )
    finally:
        await conn.close()

asyncio.run(main())
