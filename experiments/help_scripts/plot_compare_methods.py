import matplotlib.pyplot as plt
import numpy as np
# Data (only AR values)
data = {
    "AR YCB-V":
    {
        "measured": {
            "Coarse": 0.481,
            "Refined, align=False": 0.508,
            "Refined, align=True": 0.619
        },
        "reported": {
            "Coarse": 0.452,
            "Refined, align=False": 0.497,
            "Refined, align=True": 0
        }
    },
    "AR T-LESS": {
        "measured": {
            "Coarse": 0.334,
            "Refined, align=False": 0.388,
            "Refined, align=True": 0.381
        },
        "reported": {
            "Coarse": 0.338,
            "Refined, align=False": 0.396,
            "Refined, align=True": 0
        }
    },
    "AR LM-O": {
        "measured": {
            "Coarse": 0.375,
            "Refined, align=False": 0.382,
            "Refined, align=True": 0.426
        },
        "reported": {
            "Coarse": 0.397,
            "Refined, align=False": 0.395,
            "Refined, align=True": 0
        },
    },
    "AR TUD-L": {
        "measured": {
            "Coarse": 0.463,
            "Refined, align=False": 0.549,
            "Refined, align=True": 0.502
        },
        "reported": {
            "Coarse": 0.469,
            "Refined, align=False": 0.567,
            "Refined, align=True": 0
        }
    }
}

# Plot
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
axes = axes.flatten()

bar_width = 0.25
methods = ["Coarse", "Refined, align=False", "Refined, align=True"]
colors = ["#4C72B0", "#55A868"]  # measured, reported

for ax, (dataset, vals) in zip(axes, data.items()):
    measured = [vals["measured"].get(m, 0) for m in methods]
    reported = [vals["reported"].get(m, 0) for m in methods]
    x = np.arange(len(methods))

    ax.bar(x - bar_width/2, measured, width=bar_width, label="Measured", color=colors[0])
    ax.bar(x + bar_width/2, reported, width=bar_width, label="Reported", color=colors[1])

    ax.set_ylim(0, 0.7)
    ax.set_title(dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

# Shared labels + legend
fig.text(0.5, 0.04, 'Method', ha='center')
fig.text(0.04, 0.5, 'AR', va='center', rotation='vertical')
fig.legend(["Measured", "Reported"], loc="upper center", ncol=2)

plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])
plt.show()
plt.savefig("foundpose_comparison_methods.png", dpi=300, bbox_inches='tight')

