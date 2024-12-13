import os
os.chdir("./02_transform")

import pandas as pd
import json

df = pd.read_parquet("hf://datasets/potsawee/wiki_bio_gpt3_hallucination/data/evaluation-00000-of-00001-e91191b8ff41afbe.parquet")

target = {
    "gpt-3.5-turbo": "../01_inference/chatgpt/batch_67462c8ed99c81908c49f5851e20ecd5_output.jsonl",
    "gpt-4o-mini": "../01_inference/chatgpt/batch_67462c9d1d788190adf6c580d095c7e0_output.jsonl",
}

def process_batch(model: str):
    batch_dict = {}
    with open(f"{target[model]}", "r") as f:
        for line in f:
            data = json.loads(line)
            id = data['custom_id']
            r_id = id.split("-")[-3]
            s_id = id.split("-")[-2]
            c_id = id.split("-")[-1]
            res = data['response']['body']['choices'][0]['message']['content'].strip().lower().split(".")[0].split(",")[0]
            if res not in ["yes", "no"]:
                raise ValueError(f"Invalid response: {res}")
            if r_id not in batch_dict:
                batch_dict[r_id] = {}
            if s_id not in batch_dict[r_id]:
                batch_dict[r_id][s_id] = (0,0)
            if res == "no":
                batch_dict[r_id][s_id] = (batch_dict[r_id][s_id][0] + 1, batch_dict[r_id][s_id][1] + 1)
            else:
                batch_dict[r_id][s_id] = (batch_dict[r_id][s_id][0], batch_dict[r_id][s_id][1] + 1)
    ret = []
    for r_id in range(len(batch_dict)):
        S = []
        for s_id in range(len(batch_dict[str(r_id)])):
            S.append(batch_dict[str(r_id)][str(s_id)][0] / batch_dict[str(r_id)][str(s_id)][1])
        ret.append(str(S))
    return ret

for model in target:
    print(model)
    df[model] = process_batch(model)

df.to_csv('with-gpt3-gpt4.csv', index=False)