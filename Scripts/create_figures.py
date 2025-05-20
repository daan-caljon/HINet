import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import scienceplots

path_prefix ="NeighCov0"

plt.style.use('science')
scale = 4
plt.rcParams.update({
    'font.size': 10 * scale,
    'axes.titlesize': 12 * scale,
    'axes.labelsize': 10 * scale,
    'xtick.labelsize': 8 * scale,
    'ytick.labelsize': 8 * scale,
    'legend.fontsize': 9 * scale,
    'figure.titlesize': 14 * scale,
})

# 1. Load the WandB Excel file
# with neighb cov
if path_prefix == "NeighCov15":
    wandb_file = r""  # Path to the file with neighbor covariate
# without neighb cov
elif path_prefix == "NeighCov0":
    wandb_file=r"" # Path to the file without neighbor covariate
df = pd.read_csv(wandb_file)
df = df[df['State'] != 'crashed']
df = df[df['State'] != 'failed']
# print(df)
if df['alpha_range'].dtype == 'object':
    df['alpha_range'] = df['alpha_range'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ['true', '1']
    return bool(x)
#delete rows with State == Failed
# df = df[df['State'] != 'Failed']
#ensure that all the columns are the correct type
df['betaTreat2Outcome'] = df['betaTreat2Outcome'].astype(int)
df['betaNeighborTreatment2Outcome'] = df['betaNeighborTreatment2Outcome'].astype(int)
df['betaConfounding'] = df['betaConfounding'].astype(int)
df['betaCovariate2Outcome'] = df['betaCovariate2Outcome'].astype(float)
df['betaNeighborCovariate2Outcome'] = df['betaNeighborCovariate2Outcome'].astype(float)

df['homophily'] = df['homophily'].apply(to_bool)
print(df['homophily'])
# 2. Create empty container for structured data
def build_dataset(df_subset, homophily_str):
    print("subset", df_subset)
    datasets = []
    for beta_combo in [(2, 0), (0, 2), (2, 2)]:
        betaNeighbor, betaTreat = beta_combo
        df_beta = df_subset[
            (df_subset['betaNeighborTreatment2Outcome'] == betaNeighbor) &
            (df_subset['betaTreat2Outcome'] == betaTreat)
        ]
        # print("beta", df_beta)
        if df_beta.empty:
            print(f"No data for betaN={betaNeighbor}, betaT={betaTreat}. Skipping...")
            continue
        grouped_balanced = df_beta[df_beta['alpha_range'].apply(lambda x: len(x) > 1)].groupby('betaConfounding')
        grouped_unbalanced = df_beta[df_beta['alpha_range'].apply(lambda x: len(x) == 1)].groupby('betaConfounding')

        betaXT_vals = sorted(set(grouped_balanced.groups.keys()).union(grouped_unbalanced.groups.keys()))

        balancing_pehne, std_balancing_pehne = [], []
        no_pehne, std_no_pehne = [], []
        balancing_cnee, std_balancing_cnee = [], []
        no_cnee, std_no_cnee = [], []

        for betaXT in betaXT_vals: 
            if betaXT in grouped_balanced.groups:
                g_bal = grouped_balanced.get_group(betaXT)
                balancing_pehne.append(g_bal['avg_metric'].mean())
                std_balancing_pehne.append(g_bal['st_dev_test_metric'].mean())
                balancing_cnee.append(g_bal['avg_val_cf_metric'].mean())
                std_balancing_cnee.append(g_bal['st_dev_val_cf_metric'].mean())
            else:
                balancing_pehne.append(np.nan)
                std_balancing_pehne.append(np.nan)
                balancing_cnee.append(np.nan)
                std_balancing_cnee.append(np.nan)

            if betaXT in grouped_unbalanced.groups:
                g_no = grouped_unbalanced.get_group(betaXT)
                no_pehne.append(g_no['avg_metric'].mean())
                std_no_pehne.append(g_no['st_dev_test_metric'].mean())
                no_cnee.append(g_no['avg_val_cf_metric'].mean())
                std_no_cnee.append(g_no['st_dev_val_cf_metric'].mean())
            else:
                no_pehne.append(np.nan)
                std_no_pehne.append(np.nan)
                no_cnee.append(np.nan)
                std_no_cnee.append(np.nan)


        datasets.append({
            "condition": f"$\\beta_N={betaNeighbor}, \\beta_T={betaTreat}$",
            "homophily_status": homophily_str,
            "file_path_suffix": f"{homophily_str.lower().replace(' ', '_')}_betaN{betaNeighbor}_betaT{betaTreat}.pdf",
            "confounding_list": betaXT_vals,
            "balancing_PEHNE": balancing_pehne,
            "std_balancing_PEHNE": std_balancing_pehne,
            "no_PEHNE": no_pehne,
            "std_no_PEHNE": std_no_pehne,
            "balancing_CNEE": balancing_cnee,
            "std_balancing_CNEE": std_balancing_cnee,
            "no_CNEE": no_cnee,
            "std_no_CNEE": std_no_cnee,
        })
    return datasets


# 3. Split based on homophily
datasets = []
datasets += build_dataset(df[df['homophily'] == True], 'Homophily')
datasets += build_dataset(df[df['homophily'] == False], 'No Homophily')
# print(df[df['homophily'] == True])

# 4. Plotting loop
common_figsize = (11, 8)

for i, data in enumerate(datasets):
    print(f"Processing: {data['homophily_status']} - {data['condition']}")

    title_prefix = data["homophily_status"]
    file_path_suffix = data["file_path_suffix"]
    confounding_list = data["confounding_list"]

    # --- Figure 1: PEHNE Performance ---
    plt.figure(figsize=common_figsize)

    if data["balancing_PEHNE"]:
        plt.errorbar(confounding_list, data["balancing_PEHNE"], yerr=data["std_balancing_PEHNE"],
                     label="Balancing", marker='o', linestyle='-',
                     capsize=5, linewidth=2, markersize=8)
    if data["no_PEHNE"]:
        plt.errorbar(confounding_list, data["no_PEHNE"], yerr=data["std_no_PEHNE"],
                     label="$\\alpha=0$", marker='s', linestyle='-',
                     capsize=5, linewidth=2, markersize=8)

    plt.title(f"{title_prefix}", fontweight='bold')
    plt.xlabel("$\\beta_{XT}$")
    plt.ylabel("PEHNE")
    plt.xticks(confounding_list, labels=[str(c) for c in confounding_list])
    plt.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.05))
    plt.tight_layout()

    pehne_path = f"{path_prefix}_PEHNE_{file_path_suffix}"
    plt.savefig(pehne_path, bbox_inches='tight', format="pdf")
    print(f"Saved PEHNE plot to: {pehne_path}")
    plt.close()

    # --- Figure 2: CNEE Performance ---
    plt.figure(figsize=common_figsize)

    if data["balancing_CNEE"]:
        plt.errorbar(confounding_list, data["balancing_CNEE"], yerr=data["std_balancing_CNEE"],
                     label="Balancing", marker='^', linestyle='-',
                     capsize=5, linewidth=2, markersize=8)
    if data["no_CNEE"]:
        plt.errorbar(confounding_list, data["no_CNEE"], yerr=data["std_no_CNEE"],
                     label="$\\alpha=0$", marker='x', linestyle='-',
                     capsize=5, linewidth=2, markersize=8)

    plt.title(f"{title_prefix}", fontweight='bold')
    plt.xlabel("$\\beta_{XT}$")
    plt.ylabel("CNEE")
    plt.xticks(confounding_list, labels=[str(c) for c in confounding_list])
    plt.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.05))
    plt.tight_layout()

    cnee_path = f"{path_prefix}_CNEE_{file_path_suffix}"
    plt.savefig(cnee_path, bbox_inches='tight', format="pdf")
    print(f"Saved CNEE plot to: {cnee_path}")
    plt.close()

print("\n✅ All plots generated successfully.")
