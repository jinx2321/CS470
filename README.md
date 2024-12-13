# Replicating and Enhancing SelfCheckGPT for Hallucination Detection

This repository contains the implementation and experiments for replicating and extending the findings of SelfCheckGPT, a zero-resource black-box framework for hallucination detection in large language models (LLMs). The project focuses on evaluating consistency in LLM outputs and experimenting with new prompt engineering techniques and state-of-the-art LLMs.

## Project Overview

SelfCheckGPT employs five techniques for hallucination detection, with LLM Prompting demonstrating the highest efficacy. This project reproduces and validates the findings while exploring improvements in the framework. Key goals include:
- Testing SelfCheckGPT with state-of-the-art LLMs like GPT-4, LLaMA 3.2, and Mixtral-8x7B
- Enhancing LLM Prompting via prompt engineering techniques
- Evaluating detection performance for major and minor inaccuracies

## Repository Structure

#### `01_inference/`
- **`chatgpt/`:** Covers GPT-3.5-turbo and GPT-4-mini
  - `batch-gpt-3.5-turbo.jsonl`, `batch-gpt-4o-mini.jsonl`: Input files for OpenAPI execution to generate output files
  - `batch_67462c8ed99c81908c49f5851e20ecd5_output.jsonl`, `batch_67462c9d1d788190adf6c580d095c7e0_output.jsonl`: Contains output from "GPT-runs" of the SelfCheckGPT dataset 
- **`groq/`:** Covers Llama-3.2-3b-preview, Gemma2-9b-it, Mixtral-8x7b-32768
  - `process_groq.py`: Processes raw dataset to generate `.db` files
  - `Jan.db`, `janin.db`, `jay.db`, `jiwoo.db`, `overall.db`: Databases containing runs of the SelfCheckGPT dataset, covering different subsets (e.g., rows 0–48, 48–96, etc.)

#### `02_transform/`
- `batch2csv.py`: Converts `.jsonl` files into `.csv` for analysis
  - Converts `.jsonl` output files from `chatgpt/` into `with-gpt3-gpt4.csv`
- `db2csv.py`: Converts `.db` files into `.csv` format for further analysis
  - Converts `.db` files from `01_inference/` and `with-gpt3-gpt4.csv` files into `groq-hallucination.csv`

#### `03_result/`
- `evaluation.ipynb`: Notebook for analyzing and visualizing model performance, including AUC-PR scores and precision-recall curves

#### Root
- `README.md`: This documentation file

## Run

- `process_groq.py`: Contains comments explaining how to run it, please refer to it directly for guidance
- `db2csv.py, batch2csv.py`: Run via `python3 db2csv.py` or `python3 batch2csv.py` (or just `python` depending on your environment)
- `evaluation.ipynb`: Run all cells in the notebook to generate the evaluation graphs

## Results

Key findings:
- The LLM Prompting approach consistently outperforms other techniques
- Advanced models like GPT-4-mini show higher accuracy, especially for major inaccuracies
- Prompt engineering ("Rabbit Hunting") yields mixed results, requiring further refinement

## Citation
- **SelfCheckGPT Paper**: [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
