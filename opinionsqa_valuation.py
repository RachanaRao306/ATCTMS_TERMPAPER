import os
import ast
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

WAVES = ["W26", "W27", "W29","W32","W34","W36","W41","W42","W43","W45","W49","W50","W54","W82","W92"]
PERSONAS = ["USA", "Germany", "India"]
DATA_DIR = r"" #provide the directory based on LLM model  
model_name = "qwen"
# To run this all the responses in the dataset along with persona generated answer is necessary for one particular model at a time
def get_max_wd(ordinal_values):
    d0, d1 = np.zeros(len(ordinal_values)), np.zeros(len(ordinal_values))
    d0[np.argmax(ordinal_values)] = 1
    d1[np.argmin(ordinal_values)] = 1
    return wasserstein_distance(ordinal_values, ordinal_values, d0, d1)


def strip_refused(options, dist):
    if not isinstance(options, list):
        raise TypeError(f"Expected options to be a list, got {type(options)}: {options}")
    dist = np.array(dist)
    if options[-1].strip().lower() == "refused":
        clean_dist = dist[:-1]
        if clean_dist.sum() == 0:
            return None, None
        return options[:-1], clean_dist / clean_dist.sum()
    return options, dist


def build_human_distribution(responses_df, question_key, options, wave):
    weight_col = f"WEIGHT_{wave}"
    if question_key not in responses_df.columns or weight_col not in responses_df.columns:
        return None
    sub = responses_df[[question_key, weight_col]].dropna(subset=[question_key])
    weighted_counts = sub.groupby(question_key)[weight_col].sum()
    dist = np.array([weighted_counts.get(opt, 0.0) for opt in options])
    total = dist.sum()
    if total == 0:
        return None
    return dist / total


def wd_alignment(dist_a, dist_b):
    n = len(dist_a)
    if n != len(dist_b) or n < 2:
        return None
    ordinal_values = list(range(1, n + 1))
    wd = wasserstein_distance(ordinal_values, ordinal_values, dist_a, dist_b)
    return 1 - (wd / get_max_wd(ordinal_values))


representativeness_rows = []
divergence_rows = []

for wave in WAVES:
    persona_dfs = {}
    for persona in PERSONAS:
        path = os.path.join(DATA_DIR, f"Pew_American_Trends_Panel_{wave}_{persona}_{model_name}.csv")
        df = pd.read_csv(path)
        df["options"] = df["options"].apply(ast.literal_eval)
        df["probabilities"] = df["probabilities"].apply(ast.literal_eval)

        dupes = df[df.duplicated(subset="key", keep=False)]
        if not dupes.empty:
            print(f"Warning: {len(dupes)} duplicate key rows found in {path}")
            print(dupes["key"].unique())
        df = df.drop_duplicates(subset="key", keep="first")

        persona_dfs[persona] = df.set_index("key")

    responses_path = os.path.join(DATA_DIR, f"responses_{wave}.csv")
    responses_df = pd.read_csv(responses_path, low_memory=False)

    common_keys = set(persona_dfs["USA"].index)
    for p in PERSONAS[1:]:
        common_keys &= set(persona_dfs[p].index)

    for key in common_keys:
        options = persona_dfs["USA"].loc[key, "options"]

        human_dist_raw = build_human_distribution(responses_df, key, options, wave)
        usa_dist_raw = persona_dfs["USA"].loc[key, "probabilities"]

        if human_dist_raw is not None:
            clean_opts, human_clean = strip_refused(options, human_dist_raw)
            _, usa_clean = strip_refused(options, usa_dist_raw)
            if human_clean is not None and usa_clean is not None:
                score = wd_alignment(usa_clean, human_clean)
                representativeness_rows.append({
                    "wave": wave, "key": key, "persona": "USA", "alignment_vs_human": score
                })

        for persona in ["Germany", "India"]:
            other_dist_raw = persona_dfs[persona].loc[key, "probabilities"]
            _, usa_clean2 = strip_refused(options, usa_dist_raw)
            _, other_clean = strip_refused(options, other_dist_raw)
            if usa_clean2 is not None and other_clean is not None:
                div_score = wd_alignment(usa_clean2, other_clean)
                divergence_rows.append({
                    "wave": wave, "key": key,
                    "comparison": f"USA_vs_{persona}",
                    "similarity": div_score,
                    "divergence": 1 - div_score if div_score is not None else None
                })

representativeness_df = pd.DataFrame(representativeness_rows)
divergence_df = pd.DataFrame(divergence_rows)

representativeness_df.to_csv("opinionqa_representativeness.csv", index=False)
divergence_df.to_csv("opinionqa_divergence.csv", index=False)

print("=== USA persona representativeness (mean) ===")
print(representativeness_df["alignment_vs_human"].mean())

print("\n=== Divergence from USA persona, by comparison ===")
print(divergence_df.groupby("comparison")["divergence"].mean())