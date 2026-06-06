import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_benchmark_results(results: list[dict], output_dir: Path):
    df = pd.DataFrame(results)
    
    # Filter out errors for plotting
    df = df[df['wer'].notnull()]
    
    # Plot WER by model
    plt.figure(figsize=(10, 6))
    df.boxplot(column='wer', by='model')
    plt.title('Word Error Rate (WER) Distribution by Model')
    plt.suptitle('')
    plt.ylabel('WER')
    plt.savefig(output_dir / "wer_comparison.png")
    
    # Plot CER by model
    plt.figure(figsize=(10, 6))
    df.boxplot(column='cer', by='model')
    plt.title('Character Error Rate (CER) Distribution by Model')
    plt.suptitle('')
    plt.ylabel('CER')
    plt.savefig(output_dir / "cer_comparison.png")
    
    print(f"Plots saved to {output_dir}")