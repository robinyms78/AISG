from pathlib import Path
import tkinter as tk
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------------------------------------
# 1. Select numerical features
# --------------------------------------------------

# Import train and test data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"

df = pd.read_csv(DATA_DIR / "train.csv")
# Convert SalePrice to thousands
df["SalePrice_k"] = df["SalePrice"] / 1000

numerical_cols = (
    df.select_dtypes(include="number")
    .columns
    .drop("SalePrice")
    .drop("SalePrice_k")
    .drop("Id", errors="ignore")
)


# ============================================================
# 2. Calculate correlations with SalePrice
# ============================================================

correlations = (
    df[numerical_cols.tolist() + ["SalePrice"]]
    .corr()["SalePrice"]
    .drop("SalePrice")
)


# Print correlations
print("\nCorrelation with SalePrice:")
print(
    correlations
    .sort_values(ascending=False)
)

# --------------------------------------------------
# 3. Create Tkinter window
# --------------------------------------------------

root = tk.Tk()

root.title(
    "Numerical Features vs SalePrice_k"
)

root.geometry("1600x900")


# --------------------------------------------------
# 4. Create scrollable canvas
# --------------------------------------------------

canvas = tk.Canvas(root)

scrollbar = tk.Scrollbar(
    root,
    orient="vertical",
    command=canvas.yview
)

canvas.configure(
    yscrollcommand=scrollbar.set
)

scrollbar.pack(
    side="right",
    fill="y"
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)


# --------------------------------------------------
# 5. Create frame inside canvas
# --------------------------------------------------

plot_frame = tk.Frame(canvas)

canvas.create_window(
    (0, 0),
    window=plot_frame,
    anchor="nw"
)


# --------------------------------------------------
# 6. Create matplotlib figure
# --------------------------------------------------

n_features = len(numerical_cols)

# 4 plots per row
n_cols = 4

# Calculate number of rows
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(20, 4.5 * n_rows)
)

# Make axes consistently 1-dimensional
axes = axes.flatten()


# --------------------------------------------------
# 7. Create scatter + regression plot
# --------------------------------------------------

for i, feature in enumerate(numerical_cols):

    ax = axes[i]

    # Remove missing values
    plot_data = df[
        ["SalePrice_k", feature]
    ].dropna()

    # Scatter plot + linear regression
    sns.regplot(
        data=plot_data,
        x="SalePrice_k",
        y=feature,
        scatter_kws={
            "alpha": 0.4,
            "s": 20
        },
        line_kws={
            "linewidth": 2
        },
        ax=ax
    )

    # Title
    ax.set_title(
        f"{feature} vs SalePrice_k"
    )

    # Axis labels
    ax.set_xlabel(
        "SalePrice_k"
    )

    ax.set_ylabel(
        feature
    )


# --------------------------------------------------
# 8. Hide unused subplots
# --------------------------------------------------

for i in range(
    n_features,
    len(axes)
):
    axes[i].set_visible(False)


# --------------------------------------------------
# 9. Overall title
# --------------------------------------------------

fig.suptitle(
    "Numerical Features vs SalePrice_k",
    fontsize=18
)


plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)


# --------------------------------------------------
# 10. Embed matplotlib into Tkinter
# --------------------------------------------------

figure_canvas = FigureCanvasTkAgg(
    fig,
    master=plot_frame
)

figure_canvas.draw()

figure_canvas.get_tk_widget().pack()


# --------------------------------------------------
# 11. Configure scrolling
# --------------------------------------------------

plot_frame.update_idletasks()

canvas.configure(
    scrollregion=canvas.bbox("all")
)


# --------------------------------------------------
# 12. Mouse wheel scrolling
# --------------------------------------------------

def on_mousewheel(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    on_mousewheel
)


# --------------------------------------------------
# 13. Start GUI
# --------------------------------------------------

root.mainloop()