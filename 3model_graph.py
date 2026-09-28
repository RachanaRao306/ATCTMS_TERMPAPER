import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

MODEL_TAGS = ["gemma", "mistral", "qwen"]
MODEL_LABELS = {"gemma": "Gemma-2-2B-it", "mistral": "Mistral-7B-Instruct-v0.2", "qwen": "Qwen2.5-7B-Instruct"}


# FIGURE 1: OpinionQA — USA-persona Representativeness, 3 models


rows = []
for tag in MODEL_TAGS:
    rep_df = pd.read_csv(f"temp_model_comp/opinionqa_representativeness_{tag}.csv")
    rows.append({"model": MODEL_LABELS[tag], "representativeness": rep_df["alignment_vs_human"].mean()})

fig1_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=fig1_df, x="model", y="representativeness", hue="model", ax=ax, legend=False)
ax.set_ylabel("Representativeness (USA persona vs. Human)")
ax.set_xlabel("Model")
ax.set_title("OpinionQA: Representativeness by Model")
ax.set_ylim(0, 1)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("fig1_opinionqa_representativeness_3models.png", dpi=200)
plt.close(fig)


# FIGURE 2: GlobalOpinionQA — Representativeness vs. Own Country, 3 models x 3 personas


rows = []
for tag in MODEL_TAGS:
    rep_df = pd.read_csv(f"temp_model_comp/globalopinionqa_representativeness_{tag}.csv")
    by_persona = rep_df.groupby("persona")["alignment_vs_own_country"].mean()
    for persona in ["usa", "germany", "india"]:
        rows.append({
            "model": MODEL_LABELS[tag],
            "persona": persona,
            "representativeness": by_persona.get(persona)
        })

fig2_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=fig2_df, x="persona", y="representativeness", hue="model", ax=ax)
ax.set_ylabel("Representativeness vs. Own Country")
ax.set_xlabel("Persona")
ax.set_title("GlobalOpinionQA: Representativeness by Model and Persona")
ax.set_ylim(0, 1)
ax.legend(title="Model", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("fig2_globalopinionqa_representativeness_3models.png", dpi=200)
plt.close(fig)


# FIGURE 3: GlobalOpinionQA — Pairwise Divergence, 3 models


rows = []
for tag in MODEL_TAGS:
    div_df = pd.read_csv(f"temp_model_comp/globalopinionqa_divergence_{tag}.csv")
    by_pair = div_df.groupby("comparison")["divergence"].mean()
    for pair in ["usa_vs_germany", "usa_vs_india", "germany_vs_india"]:
        rows.append({
            "model": MODEL_LABELS[tag],
            "comparison": pair,
            "divergence": by_pair.get(pair)
        })

fig3_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=fig3_df, x="comparison", y="divergence", hue="model", ax=ax)
ax.set_ylabel("Divergence")
ax.set_xlabel("Persona Pair")
ax.set_title("GlobalOpinionQA: Pairwise Divergence by Model")
ax.legend(title="Model", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("fig3_globalopinionqa_divergence_3models.png", dpi=200)
plt.close(fig)


# FIGURE 4: Combined overview — Representativeness vs. Divergence, all 3 models


rep_rows = []
for tag in MODEL_TAGS:
    rep_df = pd.read_csv(f"temp_model_comp/globalopinionqa_representativeness_{tag}.csv")
    overall_rep = rep_df["alignment_vs_own_country"].mean()
    rep_rows.append({"model": MODEL_LABELS[tag], "metric": "Representativeness (mean)", "score": overall_rep})

div_rows = []
for tag in MODEL_TAGS:
    div_df = pd.read_csv(f"temp_model_comp/globalopinionqa_divergence_{tag}.csv")
    overall_div = div_df["divergence"].mean()
    div_rows.append({"model": MODEL_LABELS[tag], "metric": "Divergence (mean)", "score": overall_div})

fig4_df = pd.DataFrame(rep_rows + div_rows)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=fig4_df, x="model", y="score", hue="metric", ax=ax)
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("GlobalOpinionQA: Representativeness vs. Divergence Overview (All Models)")
ax.legend(title="Metric", loc="upper right", fontsize=8)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("fig4_globalopinionqa_combined_3models.png", dpi=200)
plt.close(fig)


# Save underlying summary tables (for cross-checking numbers in your paper)


fig1_df.to_csv("summary_fig1_opinionqa_representativeness_3models.csv", index=False)
fig2_df.to_csv("summary_fig2_globalopinionqa_representativeness_3models.csv", index=False)
fig3_df.to_csv("summary_fig3_globalopinionqa_divergence_3models.csv", index=False)
fig4_df.to_csv("summary_fig4_globalopinionqa_combined_3models.csv", index=False)

print("All 3-model comparison figures and summary tables saved.")