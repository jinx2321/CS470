# Replicating and Enhancing SelfCheckGPT for Hallucination Detection

This repository contains the implementation and experiments for replicating and extending the findings of SelfCheckGPT, a zero-resource black-box framework for hallucination detection in large language models (LLMs). The project focuses on evaluating consistency in LLM outputs and experimenting with new prompt engineering techniques and state-of-the-art LLMs.

## Project Overview

SelfCheckGPT employs five techniques for hallucination detection, with LLM Prompting demonstrating the highest efficacy. This project reproduces and validates the findings while exploring improvements in the framework. Key goals include:
- Testing SelfCheckGPT with state-of-the-art LLMs like GPT-4, LLaMA 3.2, and Mixtral-8x7B
- Enhancing LLM Prompting via prompt engineering techniques
- Evaluating detection performance for major and minor inaccuracies

## Repository Structure

### Database Files
- `Jan.db`, `janin.db`, `jay.db`, `jiwoo.db`: These files contain runs of the SelfCheckGPT dataset, covering different subsets (e.g., rows 0–48, 48–96, etc.)
  
### CSV Files
- `groq-hallucination.csv`: Contains processed hallucination data for evaluation
- `with-gpt3-gpt4.csv`: Stores performance metrics comparing GPT-3.5-turbo and GPT-4-mini models

### Python Scripts
- `db2csv.py`: Converts `.db` files into `.csv` for further analysis
- `process_groq.py`: Processes raw data and supports prompt parsing improvements
- `utils.py`: Contains utility functions for data handling and visualization

### Jupyter Notebook
- `evaluation.ipynb`: Notebook for analyzing and visualizing model performance, including AUC-PR scores and precision-recall curves

## Run

- `process_groq.py`: Contains comments explaining how to run it, please refer to it directly for guidance.

- `evaluation.ipynb`: Run all cells in the notebook to generate the evaluation graphs.

## Results

Key findings:
- The LLM Prompting approach consistently outperforms other techniques
- Advanced models like GPT-4-mini show higher accuracy, especially for major inaccuracies
- Prompt engineering ("Rabbit Hunting") yields mixed results, requiring further refinement

## Citation
- **SelfCheckGPT Paper**: [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
