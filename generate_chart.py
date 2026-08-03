import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set ggplot style to match the sample
plt.style.use('ggplot')

# Data from the Kaggle screenshots
data = {
    'Fold': ['Video 1', 'Video 1', 'Video 3', 'Video 3', 'Video 4 (Plain)', 'Video 4 (Plain)'],
    'Metric': ['Accuracy', 'Macro F1', 'Accuracy', 'Macro F1', 'Accuracy', 'Macro F1'],
    'Value': [0.333, 0.117, 0.656, 0.421, 0.454, 0.308]
}
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8, 6))

# Custom colors matching the ggplot2 sample (coral and cyan)
colors = ['#f8766d', '#00bfc4']

x = np.arange(len(df['Fold'].unique()))
width = 0.35

acc_vals = df[df['Metric'] == 'Accuracy']['Value'].values
f1_vals = df[df['Metric'] == 'Macro F1']['Value'].values

rects1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy', color=colors[0])
rects2 = ax.bar(x + width/2, f1_vals, width, label='Macro F1', color=colors[1])

# Add text labels on bars like the sample (with border)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # Position label in the middle of the bar
        ax.annotate(f'{height*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.5),
                    fontsize=10, color='black')

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Score')
ax.set_xlabel('Hold-out Video (LOVO Fold)')
ax.set_title('Mamba SSM Classifier Test Metrics by Fold')
ax.set_xticks(x)
ax.set_xticklabels(df['Fold'].unique())

# Legend on the right side
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(title='Metric', loc='center left', bbox_to_anchor=(1, 0.5))

# Remove grid lines for a cleaner look or keep them for ggplot style
# plt.grid(True) 

plt.savefig('lovo_metrics_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated successfully at lovo_metrics_chart.png")
