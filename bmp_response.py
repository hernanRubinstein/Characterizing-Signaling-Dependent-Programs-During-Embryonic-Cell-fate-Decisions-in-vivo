import subprocess
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.decomposition import PCA
import traceback
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def install_required_packages():
    with open('requirements.txt', 'r') as file:
        required_packages = [line.strip() for line in file if line.strip()]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"The package {package} is needed for this program to run - INSTALLING...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} has been successfully installed.")


install_required_packages()


def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index.name = None  # Remove the index name if it's set to None
        print(f"Successfully loaded {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print(f"First few rows:\n{df.head()}\n")
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        df = pd.DataFrame()
    return df


ignore_genes = load_csv("./ignore_genes.csv")
egc_atlas = load_csv("./egc_atlas.csv")
metadata_atlas = load_csv("./metadata_atlas.csv")
egc_query = load_csv("./egc_query.csv")
metadata_query = load_csv("./metadata_query.csv")

if ignore_genes.empty or egc_atlas.empty or metadata_atlas.empty or egc_query.empty or metadata_query.empty:
    print("One or more required files could not be loaded. Please check the file paths and contents.")
    sys.exit(1)

# Filter out ignore genes from the start
ignore_genes_set = set(ignore_genes.index)
egc_atlas = egc_atlas[~egc_atlas.index.isin(ignore_genes_set)]
egc_query = egc_query[~egc_query.index.isin(ignore_genes_set)]

# Process atlas data
reg_atlas = float(f"{1/egc_atlas.sum().mean():.1e}")
egc_atlasn = egc_atlas.div(egc_atlas.sum())
legc_atlasn = np.log2(egc_atlasn + reg_atlas)

# Process query data
reg_query = float(f"{1/egc_query.sum().mean():.1e}")
egcn = egc_query.div(egc_query.sum())
legcn = np.log2(egcn + reg_query)

fold_legcn = pd.DataFrame({
    "BMP4_1ug": legcn["BMP4_1ug"] - legcn["NTC"],
    "BMP4_2ug": legcn["BMP4_2ug"] - legcn["NTC"]
})

# Default clustering method
clustering_method = 'average'

genes_for_pca = None


def compute_knn_linkage(data, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    distances = pairwise_distances(data, knn.kneighbors_graph(data).toarray())
    linkage_matrix = linkage(distances, method='average')
    return linkage_matrix


def update_plot(min_var, min_exp, vmin, vmax, regulation, k_neighbors, save_heatmap=False, generate_pca=False):
    global genes_for_pca
    try:
        f_var_up = (fold_legcn.max(axis=1) > min_var)
        f_var_down = (fold_legcn.min(axis=1) < -min_var)
        f_min = (legc_atlasn.max(axis=1) > min_exp)
        
        if regulation == 'Up and Downreulated':
            genes_to_plot = fold_legcn.index[(f_var_up | f_var_down) & f_min]
            title = "Differentially Expressed Genes in Response to BMP4"
        elif regulation == 'Upreulated':
            genes_to_plot = fold_legcn.index[f_var_up & f_min]
            title = "Upregulated Genes in Response to BMP4"
        else:  # down
            genes_to_plot = fold_legcn.index[f_var_down & f_min]
            title = "Downregulated Genes in Response to BMP4"
        
        genes_for_pca = genes_to_plot
        
        legcn_norm = legcn.subtract(legcn.mean(axis=1), axis=0)
        legcn_norm = legcn_norm.loc[genes_to_plot, ["NTC", "BMP4_1ug", "BMP4_2ug"]]
        
        # Clustering
        row_linkage = linkage(legcn_norm, method='average')
        row_order = dendrogram(row_linkage, no_plot=True)['leaves']
        
        legcn_norm = legcn_norm.iloc[row_order]
        
        fig, (ax_title, ax_heatmap, ax_colorbar) = plt.subplots(nrows=3, figsize=(10, 8), 
                                                                gridspec_kw={'height_ratios': [1, 20, 0.5]})
        
        # Title
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, title, fontsize=10, ha='center', va='center')
        
        # Heatmap
        sns.heatmap(legcn_norm, ax=ax_heatmap, cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax,
                    xticklabels=["Control", "+BMP4 [1uM]", "+BMP4 [2uM]"], yticklabels=True, cbar=False)
        
        # Adjust font size for gene names and rotate column names
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontsize=4)
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), fontsize=6, ha='center')
        ax_heatmap.xaxis.tick_top()
        
        # Colorbar
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_colorbar, orientation="horizontal")
        cbar.ax.tick_params(labelsize=4)  # Adjust font size of colorbar ticks
        cbar.set_label("Log2(relative expression)", fontsize=5)  # Add title to the colorbar
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9, bottom=0.1)  # Adjust position of colorbar
        
        if save_heatmap:
            fig.savefig("heatmap.jpeg", format='jpeg')
            messagebox.showinfo("Save Heatmap", "Heatmap has been saved as 'heatmap.jpeg' in the main directory.")
        
        if generate_pca:
            if genes_for_pca is not None:
                save_pca_plot()
        
        return fig
    except Exception as e:
        print(f"Error in update_plot: {str(e)}")
        traceback.print_exc()
        return None


def save_pca_plot():
    try:
        global genes_for_pca, legc_atlasn, metadata_atlas

        # Check if genes_for_pca is not None and contains genes
        if genes_for_pca is not None and len(genes_for_pca) > 0:
            print(f"Number of genes for PCA: {len(genes_for_pca)}")

            # Ensure the genes_for_pca are in legc_atlasn
            if not set(genes_for_pca).issubset(set(legc_atlasn.index)):
                raise ValueError("Some genes in genes_for_pca are not present in legc_atlasn.")

            # Transpose legc_atlasn to apply PCA on genes
            components = PCA(n_components=3).fit_transform(legc_atlasn.loc[genes_for_pca].T)
            components_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'], index=legc_atlasn.columns)

            # Ensure metadata_atlas has color and cell_type columns
            if 'color' not in metadata_atlas.columns or 'cell_type' not in metadata_atlas.columns:
                raise ValueError("metadata_atlas does not contain 'color' or 'cell_type' columns.")

            # Validate colors in metadata_atlas
            valid_colors = metadata_atlas['color'].apply(lambda c: c if mcolors.is_color_like(c) else '#000000')
            components_df['color'] = valid_colors.values
            components_df['cell_type'] = metadata_atlas['cell_type'].values
            print(components_df)

            # Plotting the PCA components
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(components_df['PC1'], components_df['PC2'], components_df['PC3'],
                                 c=components_df['color'], s=50, alpha=0.8)

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('PCA Over BMP4 Response')

            # Add legend
            unique_colors = components_df[['color', 'cell_type']].drop_duplicates()
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type,
                                          markersize=8, markerfacecolor=color)
                               for color, cell_type in zip(unique_colors['color'], unique_colors['cell_type'])]
            ax.legend(handles=legend_elements, title="Cell Types", loc='center left', bbox_to_anchor=(1, 0.5),
                      ncol=2, fontsize='small')

            plt.savefig("WT_PCA_analysis.jpeg", format='jpeg', bbox_inches='tight')
            messagebox.showinfo("Save PCA 3D", "3D PCA plot has been saved as 'WT_PCA_analysis.jpeg' in the main directory.")
        else:
            raise ValueError("No genes to plot for the selected criteria.")
    except Exception as e:
        print(f"Error in save_pca_plot: {str(e)}")
        traceback.print_exc()


root = tk.Tk()
root.title("Gene Expression Analysis")
root.configure(bg='#2E2E2E')

style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#2E2E2E')
style.configure('TLabel', background='#2E2E2E', foreground='white')
style.configure('TScale', background='#2E2E2E', troughcolor='#4A4A4A')

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

control_frame = ttk.Frame(main_frame, padding="10")
control_frame.pack(side=tk.LEFT, fill=tk.Y)

plot_frame = ttk.Frame(main_frame, padding="10")
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

ttk.Label(control_frame, text="Heatmap configurations", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

min_var = tk.DoubleVar(value=1.0)
min_exp = tk.DoubleVar(value=-14)
vmin = tk.DoubleVar(value=-2.0)
vmax = tk.DoubleVar(value=2.0)
regulation = tk.StringVar(value='Up and Downreulated')
k_neighbors = tk.IntVar(value=5)
save_heatmap_var = tk.BooleanVar()
generate_pca_var = tk.BooleanVar()


def create_slider(parent, variable, from_, to, text):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=5)
    ttk.Label(frame, text=text).pack(anchor=tk.W)
    slider = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, length=200)
    slider.pack(fill=tk.X)
    value_label = ttk.Label(frame, text=f"[{variable.get():.2f}]")
    value_label.pack(anchor=tk.E)
    return slider, value_label


# Gene count display
gene_count_label = ttk.Label(control_frame, text=f"Gene #: {0}", font=('Arial', 10, 'bold'))
gene_count_label.pack(anchor=tk.W, pady=5)

# Create sliders
fold_change_slider, fold_change_label = create_slider(control_frame, min_var, 0, 6, "Fold change:")
min_exp_slider, min_exp_label = create_slider(control_frame, min_exp, -16, -10, "Minimal expression:")
vmin_slider, vmin_label = create_slider(control_frame, vmin, -10, 0, "Minimal border:")
vmax_slider, vmax_label = create_slider(control_frame, vmax, 0, 10, "Maximal border:")

# Regulation dropdown
reg_frame = ttk.Frame(control_frame)
reg_frame.pack(fill=tk.X, pady=5)
ttk.Label(reg_frame, text="Regulation:").pack(anchor=tk.W)
regulation_dropdown = ttk.Combobox(reg_frame, textvariable=regulation, values=['Up and Downreulated', 'Upreulated', 'Downreulated'])
regulation_dropdown.pack(fill=tk.X)

# Number of Neighbors (K) slider
kn_slider_frame = ttk.Frame(control_frame)
kn_slider, kn_label = create_slider(kn_slider_frame, k_neighbors, 1, 20, "Number of Neighbors (K):")

# Save heatmap checkbox
save_heatmap_checkbox = ttk.Checkbutton(control_frame, text="Save heatmap", variable=save_heatmap_var)
save_heatmap_checkbox.pack(anchor=tk.W, pady=5)

# Generate PCA checkbox
generate_pca_checkbox = ttk.Checkbutton(control_frame, text="Generate PCA", variable=generate_pca_var)
generate_pca_checkbox.pack(anchor=tk.W, pady=5)

def update(*args):
    fig = update_plot(min_var.get(), min_exp.get(), vmin.get(), vmax.get(), regulation.get(), k_neighbors.get(), save_heatmap_var.get(), generate_pca_var.get())
    if fig is not None:
        canvas.figure = fig
        canvas.draw()

    fold_change_label.config(text=f"[{min_var.get():.2f}]")
    min_exp_label.config(text=f"[{min_exp.get():.2f}]")
    vmin_label.config(text=f"[{vmin.get():.2f}]")
    vmax_label.config(text=f"[{vmax.get():.2f}]")
    kn_label.config(text=f"[{k_neighbors.get()}]")
    gene_count_label.config(text=f"Gene #: {len(genes_for_pca) if genes_for_pca is not None else 0}")

min_var.trace("w", update)
min_exp.trace("w", update)
vmin.trace("w", update)
vmax.trace("w", update)
regulation.trace("w", update)
k_neighbors.trace("w", update)
save_heatmap_var.trace("w", update)
generate_pca_var.trace("w", update)

fig = update_plot(min_var.get(), min_exp.get(), vmin.get(), vmax.get(), regulation.get(), k_neighbors.get(), save_heatmap_var.get(), generate_pca_var.get())
if fig is not None:
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

# Add explanation section
explanation_frame = ttk.Frame(control_frame, padding="10")
explanation_frame.pack(side=tk.BOTTOM, fill=tk.X)

explanation_text = (
    "OVERVIEW:\n"
    "Gene #: Displays the number of genes currently displayed in the heatmap.\n"
    "Fold change: Controls the threshold for fold change of gene expression.\n"
    "Minimal expression: Sets the minimum expression level for genes to be included.\n"
    "Minimal border: Minimum value for the heatmap color scale.\n"
    "Maximal border: Maximum value for the heatmap color scale.\n"
    "Regulation: Selects the type of gene regulation to display.\n"
    "Save heatmap: saves a .jpeg of the current heatmap display.\n"
    "Generate PCA: generate a PCA of the wildtype atlas over the current genes display and saves a .jpeg."
)

explanation_label = ttk.Label(explanation_frame, text=explanation_text, font=('Arial', 8), background='#2E2E2E', foreground='white')
explanation_label.pack(fill=tk.X)

try:
    root.mainloop()
except Exception as e:
    print(f"Error in mainloop: {str(e)}")
    traceback.print_exc()
