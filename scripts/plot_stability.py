"""Plot locally generated bootstrap comparisons; the plot is not distributed."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root=Path(__file__).resolve().parents[1]
directory=root/"results/stability_comparison"
table=pd.read_csv(directory/"all_method_stability_summary_CI95.csv")
fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
for ax,metric in zip(axes.flat,["ARI","NMI","Mean_Jaccard","Min_Jaccard"]):
    rows=table.loc[table.metric==metric].reset_index(drop=True)
    for i,row in rows.iterrows():
        color="tab:red" if row.method=="Proposed" else "gray"
        ax.plot([row.CI95_lower,row.CI95_upper],[i,i],color=color)
        ax.scatter(row["median"],i,color=color)
    ax.set_yticks(range(len(rows)),rows.method)
    ax.set(xlim=(0,1),title=metric,xlabel="Median and 95% percentile interval")
fig.savefig(directory/"stability_comparison.png",dpi=300)
plt.close(fig)
