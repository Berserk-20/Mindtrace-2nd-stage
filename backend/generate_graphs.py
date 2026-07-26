import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# Paths
artifact_dir = r"C:\Users\sanka\.gemini\antigravity-ide\brain\03df11e1-8e93-4987-b7a0-7a94bfd66d2b\scratch"
os.makedirs(artifact_dir, exist_ok=True)

models = {
    'ResNet18 (Ours)': 'training_log_resnet18_rafdb.csv',
    'MobileNetV2': 'training_log_mobilenetv2_rafdb.csv',
    'VGG16': 'training_log_vgg16_rafdb.csv'
}

data = {}
best_metrics = []

for name, file in models.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        data[name] = df
        
        # Find best row based on Val Acc
        best_row = df.loc[df['Val Acc'].idxmax()]
        
        best_metrics.append({
            'Model': name,
            'Accuracy (%)': best_row['Val Acc'],
            'Train Acc (%)': best_row['Train Acc'],
            'Train Loss': best_row['Train Loss'],
            'Val Loss': best_row['Val Loss'],
            'Val F1 (%)': best_row['Val F1']
        })
    else:
        print(f"Warning: {file} not found.")

# Create comparison table
if best_metrics:
    metrics_df = pd.DataFrame(best_metrics)
    print("\n--- PERFORMANCE MATRIX FOR PPT ---")
    print("| Model | Accuracy (%) | Train Acc (%) | Train Loss | Val Loss | Val F1 (%) |")
    print("|---|---|---|---|---|---|")
    for idx, row in metrics_df.iterrows():
        print(f"| {row['Model']} | {row['Accuracy (%)']:.2f} | {row['Train Acc (%)']:.2f} | {row['Train Loss']:.4f} | {row['Val Loss']:.4f} | {row['Val F1 (%)']:.2f} |")

# Plot 1: Validation Accuracy over Epochs
plt.figure(figsize=(10, 6))
for name, df in data.items():
    plt.plot(df['Epoch'], df['Val Acc'], label=name, linewidth=2)
plt.title('Validation Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
acc_plot_path = os.path.join(artifact_dir, 'val_accuracy_comparison.png')
plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Validation Loss over Epochs
plt.figure(figsize=(10, 6))
for name, df in data.items():
    plt.plot(df['Epoch'], df['Val Loss'], label=name, linewidth=2)
plt.title('Validation Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
loss_plot_path = os.path.join(artifact_dir, 'val_loss_comparison.png')
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Bar chart of Best Accuracy
if best_metrics:
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics_df['Model'], metrics_df['Accuracy (%)'], width=0.4, color=['#4C72B0', '#DD8452', '#55A868'])
    plt.title('Best Validation Accuracy by Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom', fontweight='bold')

    bar_plot_path = os.path.join(artifact_dir, 'best_accuracy_bar.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nGraphs saved to: {artifact_dir}")
