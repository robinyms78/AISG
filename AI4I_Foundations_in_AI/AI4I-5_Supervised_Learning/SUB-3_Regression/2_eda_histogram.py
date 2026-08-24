from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# Import train and test data
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")


# ============================================================
# Check shape
# ============================================================

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# Check number of numerical and categorical features
# ============================================================

num_features = train_df.select_dtypes(
    include="number"
).shape[1]

cat_features = train_df.select_dtypes(
    include="object"
).shape[1]

print("Numerical features:", num_features)
print("Categorical features:", cat_features)

# ============================================================
# Check missing values for numerical and categorical features
# ============================================================

missing_num_features = train_df.select_dtypes(
    include="number").isna().sum().values

missing_cat_features = train_df.select_dtypes(
    include="object").isna().sum().values

print("Missing Numerical features:", missing_num_features)
print("Missing Categorical features:", missing_cat_features)

# ============================================================
# Select numerical columns
# ============================================================

numerical_cols = (
    train_df
    .select_dtypes(include="number")
    .columns
    .drop("Id")
    .drop("SalePrice")
)

n_features = len(numerical_cols)


# ============================================================
# Plot configuration
# ============================================================

# 8 columns:
# Histogram | Boxplot | Histogram | Boxplot |
# Histogram | Boxplot | Histogram | Boxplot

n_plot_cols = 8

# 4 numerical features per row
features_per_row = 4

# Calculate number of rows
n_rows = (n_features + features_per_row - 1) // features_per_row


# ============================================================
# Create Tkinter window
# ============================================================

root = tk.Tk()

root.title(
    "Numerical Feature Distributions"
)

root.geometry(
    "1800x900"
)


# ============================================================
# Create canvas and scrollbar
# ============================================================

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


# ============================================================
# Frame inside canvas
# ============================================================

plot_frame = tk.Frame(canvas)

canvas.create_window(
    (0, 0),
    window=plot_frame,
    anchor="nw"
)


# ============================================================
# Create Matplotlib figure
# ============================================================

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_plot_cols,
    figsize=(24, 4 * n_rows)
)


# Make axes consistently 2D
axes = axes.reshape(
    n_rows,
    n_plot_cols
)


# ============================================================
# Generate plots
# ============================================================

for i, col in enumerate(numerical_cols):

    # --------------------------------------------------------
    # Determine row
    # --------------------------------------------------------

    row = i // features_per_row

    # --------------------------------------------------------
    # Determine starting column
    #
    # Feature 0 -> columns 0, 1
    # Feature 1 -> columns 2, 3
    # Feature 2 -> columns 4, 5
    # Feature 3 -> columns 6, 7
    # --------------------------------------------------------

    col_start = (i % features_per_row) * 2


    # ========================================================
    # Histogram
    # ========================================================

    sns.histplot(
        train_df[col].dropna(),
        bins=30,
        kde=True,
        ax=axes[row, col_start]
    )

    axes[row, col_start].set_title(
        f"Histogram: {col}"
    )

    axes[row, col_start].set_xlabel(
        col
    )

    axes[row, col_start].set_ylabel(
        "Frequency"
    )


    # ========================================================
    # Boxplot
    # ========================================================

    sns.boxplot(
        x=train_df[col].dropna(),
        ax=axes[row, col_start + 1]
    )

    axes[row, col_start + 1].set_title(
        f"Boxplot: {col}"
    )

    axes[row, col_start + 1].set_xlabel(
        col
    )


# ============================================================
# Hide unused axes
# ============================================================

for ax in axes.flat:

    if not ax.has_data():
        ax.set_visible(False)


# ============================================================
# Adjust layout
# ============================================================

plt.tight_layout()


# ============================================================
# Embed Matplotlib figure into Tkinter
# ============================================================

figure_canvas = FigureCanvasTkAgg(
    fig,
    master=plot_frame
)

figure_canvas.draw()

figure_canvas.get_tk_widget().pack()


# ============================================================
# Update scrollable region
# ============================================================

plot_frame.update_idletasks()

canvas.configure(
    scrollregion=canvas.bbox("all")
)


# ============================================================
# Mouse wheel scrolling
# ============================================================

def on_mousewheel(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    on_mousewheel
)


# ============================================================
# Start window
# ============================================================

root.mainloop()