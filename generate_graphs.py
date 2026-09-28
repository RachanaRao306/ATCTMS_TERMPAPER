import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

# ---------- Load result files ----------
opqa_rep = pd.read_csv("opinionqa_representativeness.csv")
opqa_div = pd.read_csv("opinionqa_divergence.csv")
gopqa_rep = pd.read_csv("globalopinionqa_representativeness.csv")
gopqa_div = pd.read_csv("globalopinionqa_divergence.csv")


# FIGURE A: OpinionQA — representativeness + divergence, grouped by wave


rep_summary = opqa_rep.groupby("wave", as_index=False)["alignment_vs_human"].mean()
rep_summary["metric"] = "USA vs Human (Representativeness)"
rep_summary = rep_summary.rename(columns={"alignment_vs_human": "score"})

div_summary = opqa_div.groupby(["wave", "comparison"], as_index=False)["divergence"].mean()
div_summary = div_summary.rename(columns={"comparison": "metric", "divergence": "score"})

combined_opqa = pd.concat([
    rep_summary[["wave", "metric", "score"]],
    div_summary[["wave", "metric", "score"]]
], ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=combined_opqa, x="wave", y="score", hue="metric", ax=ax)
ax.set_ylabel("Score")
ax.set_xlabel("Survey Wave")
ax.set_title("OpinionQA: Representativeness and Persona Divergence by Wave")
ax.legend(title="Metric", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("figA_opinionqa_combined.png", dpi=200)
plt.close(fig)


# FIGURE B: GlobalOpinionQA — representativeness by persona/country


fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=gopqa_rep, x="persona", y="alignment_vs_own_country", hue="persona", ax=ax, legend=False)
ax.set_ylabel("Representativeness vs. Own Country")
ax.set_xlabel("Persona")
ax.set_title("GlobalOpinionQA: Persona Alignment with Own Country's Real Data")
plt.tight_layout()
plt.savefig("figB_globalopinionqa_representativeness.png", dpi=200)
plt.close(fig)


# FIGURE C: GlobalOpinionQA — pairwise divergence between personas


fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=gopqa_div, x="comparison", y="divergence", hue="comparison", ax=ax, legend=False)
ax.set_ylabel("Divergence")
ax.set_xlabel("Persona Pair")
ax.set_title("GlobalOpinionQA: Pairwise Persona Divergence")
plt.tight_layout()
plt.savefig("figC_globalopinionqa_divergence.png", dpi=200)
plt.close(fig)


# FIGURE D: GlobalOpinionQA — representativeness + divergence combined


rep_g = gopqa_rep.groupby("persona", as_index=False)["alignment_vs_own_country"].mean()
rep_g["type"] = "Representativeness"
rep_g = rep_g.rename(columns={"persona": "group", "alignment_vs_own_country": "score"})

div_g = gopqa_div.groupby("comparison", as_index=False)["divergence"].mean()
div_g["type"] = "Divergence"
div_g = div_g.rename(columns={"comparison": "group", "divergence": "score"})

combined_gopqa = pd.concat([rep_g[["group", "type", "score"]], div_g[["group", "type", "score"]]], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=combined_gopqa, x="group", y="score", hue="type", ax=ax)
ax.set_ylabel("Score")
ax.set_xlabel("Persona / Comparison")
ax.set_title("GlobalOpinionQA: Representativeness vs. Divergence Overview")
ax.legend(title="Metric Type", loc="upper right", fontsize=8)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figD_globalopinionqa_combined.png", dpi=200)
plt.close(fig)


# Summary tables for paper


combined_opqa.to_csv("summary_figA_opinionqa.csv", index=False)
gopqa_rep.groupby("persona")["alignment_vs_own_country"].agg(["mean", "std", "count"]).to_csv("summary_figB.csv")
gopqa_div.groupby("comparison")["divergence"].agg(["mean", "std", "count"]).to_csv("summary_figC.csv")
combined_gopqa.to_csv("summary_figD_globalopinionqa.csv", index=False)

print("All figures and summary tables saved.")