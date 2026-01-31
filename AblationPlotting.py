import matplotlib.pyplot as plt
import pandas as pd

def plot_rank_ablation():
    # Data from Table 2 in the report
    data = {
        'Rank': [8, 16, 32, 64],
        'LoRA_PPL': [5.55, 5.12, 5.08, 5.05],
        'DoRA_PPL': [5.30, 4.95, 4.91, 4.89]
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df['Rank'], df['LoRA_PPL'], marker='o', label='LoRA PPL', linestyle='--', color='#1f77b4')
    plt.plot(df['Rank'], df['DoRA_PPL'], marker='s', label='DoRA PPL', linewidth=2.5, color='#ff7f0e')
    
    # Annotate the gap at Rank 8
    plt.annotate('Significant Gap (+4.5%)', xy=(8, 5.4), xytext=(12, 5.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title('Impact of Rank on Validation Perplexity (Lower is Better)')
    plt.xlabel('Rank (r)')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('rank_ablation.png')
    print("Ablation plot generated.")

if __name__ == "__main__":
    plot_rank_ablation()
