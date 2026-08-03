import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set ggplot style to match the sample
plt.style.use('ggplot')

# Data based on the training split distribution
# Train: 202 | Val: 56 | Test: 27
categories = ['Coordinated\nDefense', 'Coordinated\nAttack', 'Spacing\nBreakdown', 'Delayed\nSupport']
train_counts = [95, 63, 32, 12]
test_counts = [13, 9, 4, 1]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Colors: Coral for Test, Cyan for Train (matching the sample image exactly)
color_test = '#f8766d'
color_train = '#00bfc4'

# In the sample, Test is on the left, Train is on the right for each group
rects_test = ax.bar(x - width/2, test_counts, width, label='Test', color=color_test)
rects_train = ax.bar(x + width/2, train_counts, width, label='Train', color=color_train)

total_test = sum(test_counts)

# Add text labels on Test bars just like the sample
def autolabel(rects, total):
    for rect in rects:
        height = rect.get_height()
        pct = (height / total) * 100
        # Position label in the middle of the bar
        ax.annotate(f'{pct:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc=color_test, ec="black", lw=1, alpha=0.9),
                    fontsize=10, color='black')

autolabel(rects_test, total_test)

ax.set_ylabel('count')
ax.set_xlabel('Strategy')
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Legend on the right side
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(title='Set', loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('train_test_split_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated successfully at train_test_split_chart.png")
