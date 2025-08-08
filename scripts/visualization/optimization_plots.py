import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = 'optimization_results'
CSV_FILENAME = 'hbt_optimization_results.csv'
PLOT_FILENAME = 'hbt_optimization_progress.png'

def create_analysis_plots(csv_path, plot_dir):
    """Generate analysis plots from the optimization results CSV."""
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Read the CSV
    results_df = pd.read_csv(csv_path)
    if results_df.empty:
        print("CSV file is empty.")
        return

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Parameters to plot (excluding list-based and complex parameters)
    plot_params = ['epochs', 'validation_split', 'outlier_cutoff', 'num_conv2d_layers', 'num_dense_layers']

    # Generate scatter plots for each parameter vs MAPE
    for param in plot_params:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=results_df,
            x=param,
            y='mape',
            hue='generation',
            palette='cividis_r',
            size='generation',
            sizes = (200, 50),  # Older generations get larger points,
            alpha=0.6  # Add transparency to make overlapping points visible
        )
        plt.xlabel(param.replace('_', ' ').title())
        plt.ylabel('MAPE (%)')
        plt.title(f'MAPE vs {param.replace("_", " ").title()} by Generation')
        plt.grid(True)
        plt.legend(loc='upper left')  # Place legend in top-left corner
        plt.savefig(os.path.join(plot_dir, f'mape_vs_{param}.png'))
        plt.close()

    # Generate optimization progress plot
    mape_history = results_df.groupby('generation')['mape'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(mape_history['generation'], mape_history['mape'], '-o')
    plt.xlabel('Generation')
    plt.ylabel('Average MAPE (%)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, PLOT_FILENAME))
    plt.close()

    print(f"Plots saved in: {plot_dir}")
def find_valid_run_dir(output_dir):
    """Find the most recent run directory with a valid CSV file."""
    run_dirs = [d for d in os.listdir(output_dir) if d.startswith('run_') and os.path.isdir(os.path.join(output_dir, d))]
    if not run_dirs:
        print(f"No run directories found in {output_dir}")
        return None

    # Sort directories by creation time, most recent first
    run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)), reverse=True)

    # Check each directory for a valid CSV
    for run_dir in run_dirs:
        csv_path = os.path.join(output_dir, run_dir, CSV_FILENAME)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    print(f"Using run directory: {run_dir}")
                    return os.path.join(output_dir, run_dir)
            except Exception as e:
                print(f"Error reading CSV in {run_dir}: {e}")
                continue
    print(f"No valid CSV files found in any run directory in {output_dir}")
    return None

if __name__ == "__main__":
    # Find the latest run directory with a valid CSV
    latest_run_dir = find_valid_run_dir(OUTPUT_DIR)
    if latest_run_dir:
        csv_path = os.path.join(latest_run_dir, CSV_FILENAME)
        plot_dir = os.path.join(latest_run_dir, 'plot_analysis')
        create_analysis_plots(csv_path, plot_dir)
    else:
        print("No valid run directory with a non-empty CSV found.")