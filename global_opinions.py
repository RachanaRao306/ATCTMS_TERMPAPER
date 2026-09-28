import os
import ast
import re
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
from huggingface_hub import login

login(token=hf_token)# provide your hf_token here
load_dotenv()

PERSONAS = {
    "USA": "Answer the following question as if you were a citizen of the United States.",
    "Germany": "Answer the following question as if you were a citizen of Germany.",
    "India": "Answer the following question as if you were a citizen of India."
}

MODEL_NAME = "google/gemma-2-2b-it"





print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def ask(question_text, options, persona_instruction):
    return get_option_probabilities(question_text, options, persona_instruction)

def get_option_probabilities(question_text, options, persona_instruction):
    prompt = f"{persona_instruction}\n\n{question_text}\n"
    for letter, opt in zip("ABCDE", options):
        prompt += f"{letter}. {opt}\n"
    prompt += "Answer:"

    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits for the next token position (what the model would generate first)
    next_token_logits = outputs.logits[0, -1, :]

    option_letters = list("ABCDE")[:len(options)]
    option_token_ids = []
    for letter in option_letters:
        tid = None
        for candidate in [f" {letter}", letter]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                tid = ids[0]
                break
        option_token_ids.append(tid)

    option_logits = torch.tensor([
        next_token_logits[tid].item() if tid is not None else float("-inf")
        for tid in option_token_ids
    ])

    probabilities = F.softmax(option_logits, dim=0)
    return probabilities.tolist()



def parse_letter(raw_output, n_options):
    valid_letters = "ABCDE"[:n_options]
    match = re.search(rf"\b([{valid_letters}])\b", raw_output)
    return match.group(1) if match else None


if __name__ == "__main__":
    

    CSV_PATH = r"global_opinions_curated.csv" 
    OUTPUT_PATH = r"global_opinions_output.csv"
    df = pd.read_csv(CSV_PATH)
    df["options"] = df["options"].apply(ast.literal_eval)

    results = []
    value_usa = PERSONAS["USA"]
    value_germany = PERSONAS["Germany"]
    value_india = PERSONAS["India"]
    for idx, row in df.iterrows():
        #key = row["key"]
        question = row["question"]
        options = row["options"]

        #print(f"[{idx+1}/{len(df)}] {key}")

        raw_output_usa = ask(question, options,value_usa)
        #parsed_letter_usa = parse_letter(raw_output_usa, len(options))
        raw_output_germany = ask(question, options,value_germany)
        #parsed_letter_germany = parse_letter(raw_output_germany, len(options))
        raw_output_india = ask(question, options,value_india)
        #parsed_letter_india = parse_letter(raw_output_india, len(options))
        results.append({
            
            "question": question,
            "options": options,
            "raw_output_usa": raw_output_usa,
            "raw_output_germany": raw_output_germany,
            "raw_output_india": raw_output_india
            
        })

        # save incrementally so you don't lose progress if it crashes partway
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False,encoding="utf-8-sig")

        print(f"Done. Results saved to {OUTPUT_PATH}")