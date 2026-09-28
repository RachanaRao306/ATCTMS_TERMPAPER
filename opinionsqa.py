import os
import ast
import re
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
from huggingface_hub import login

login(token=hf_token) #provide the hf_token here
load_dotenv()

PERSONAS = {
    "USA": "Answer the following question as if you were a citizen of the United States.",
    "Germany": "Answer the following question as if you were a citizen of Germany.",
    "India": "Answer the following question as if you were a citizen of India."
}

MODEL_NAME = "google/gemma-2-2b-it"
model_name = "gemma"

def build_output_path(filepath, suffix):
    directory, filename = os.path.split(filepath)

    name, ext = os.path.splitext(filename)
    output_directory = r"" #provide your directory
    return os.path.join(output_directory, f"{name}_{suffix}_{model_name}{ext}")


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


def ask(question_text, options,persona_key):
    persona_instruction = persona_key
    prompt = f"{persona_instruction}\n\n{question_text}\n"
    for letter, opt in zip("ABCDE", options):
        prompt += f"{letter}. {opt}\n"
    prompt += "Answer with just the letter of your choice."

    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer_text.strip()
def get_option_probabilities(question_text, options, persona_instruction):
    prompt = ""
    if persona_instruction:
        prompt += f"{persona_instruction}\n\n"
    prompt += f"{question_text}\n"
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

    # logits for the NEXT token (i.e. what the model would generate first)
    next_token_logits = outputs.logits[0, -1, :]

    # get token ids for "A", "B", "C", ... (try both with and without leading space)
    option_letters = list("ABCDE")[:len(options)]
    option_token_ids = []
    for letter in option_letters:
        candidates = [f" {letter}", letter]
        tid = None
        for c in candidates:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                tid = ids[0]
                break
        option_token_ids.append(tid)

    option_logits = torch.tensor([
        next_token_logits[tid].item() if tid is not None else float("-inf")
        for tid in option_token_ids
    ])

    probs = F.softmax(option_logits, dim=0)
    return probs.tolist()

def parse_letter(raw_output, n_options):
    valid_letters = "ABCDE"[:n_options]
    match = re.search(rf"\b([{valid_letters}])\b", raw_output)
    return match.group(1) if match else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate persona-specific output filenames")
    parser.add_argument("filepath", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    CSV_PATH = args.filepath
    df = pd.read_csv(CSV_PATH,sep = '\t')
    df["options"] = df["options"].apply(ast.literal_eval)
    for key,value in PERSONAS.items():
        results = []
        OUTPUT_PATH = build_output_path(CSV_PATH,key)
        for idx, row in df.iterrows():
            key = row["key"]
            question = row["question"]
            options = row["options"]

            print(f"[{idx+1}/{len(df)}] {key}")

            probs = get_option_probabilities(row["question"], options, value)
            results.append({
                "key": row["key"],
                "question": row["question"],
                "options": options,
                "probabilities": probs
            })

            # save incrementally so you don't lose progress if it crashes partway
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

            print(f"Done. Results saved to {OUTPUT_PATH}")