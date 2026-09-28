import pandas as pd

MODEL_TAGS = ["gemma", "mistral", "qwen"]

# ---------- Table 1: OpinionQA ----------
opqa_rows = []
for tag in MODEL_TAGS:
    rep_df = pd.read_csv(f"temp_model_comp/opinionqa_representativeness_{tag}.csv")
    div_df = pd.read_csv(f"temp_model_comp/opinionqa_divergence_{tag}.csv")

    rep_mean = rep_df["alignment_vs_human"].mean()
    div_germany = div_df[div_df["comparison"] == "USA_vs_Germany"]["divergence"].mean()
    div_india = div_df[div_df["comparison"] == "USA_vs_India"]["divergence"].mean()

    opqa_rows.append({
        "model": tag,
        "representativeness": rep_mean,
        "usa_vs_germany_divergence": div_germany,
        "usa_vs_india_divergence": div_india
    })

opqa_table = pd.DataFrame(opqa_rows)
opqa_table.to_csv("temp_model_comp/table1_opinionqa_model_comparison.csv", index=False)
print(opqa_table)

# ---------- Table 2: GlobalOpinionQA ----------
gopqa_rows = []
for tag in MODEL_TAGS:
    rep_df = pd.read_csv(f"temp_model_comp/globalopinionqa_representativeness_{tag}.csv")
    div_df = pd.read_csv(f"temp_model_comp/globalopinionqa_divergence_{tag}.csv")

    rep_by_persona = rep_df.groupby("persona")["alignment_vs_own_country"].mean()
    div_by_pair = div_df.groupby("comparison")["divergence"].mean()

    gopqa_rows.append({
        "model": tag,
        "rep_usa": rep_by_persona.get("usa"),
        "rep_germany": rep_by_persona.get("germany"),
        "rep_india": rep_by_persona.get("india"),
        "div_usa_germany": div_by_pair.get("usa_vs_germany"),
        "div_usa_india": div_by_pair.get("usa_vs_india"),
        "div_germany_india": div_by_pair.get("germany_vs_india"),
    })

gopqa_table = pd.DataFrame(gopqa_rows)
gopqa_table.to_csv("temp_model_comp/table2_globalopinionqa_model_comparison.csv", index=False)
print(gopqa_table)