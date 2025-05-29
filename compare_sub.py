import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the CSV files
df1 = pd.read_csv('submission (1).csv')
df2 = pd.read_csv('submission.csv')

# Compute the quotient of the 'views' columns
quotient =np.log(df1['views'] / df2['views'])

# Plot the histogram
plt.figure(figsize=(8, 4))
plt.hist(quotient, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Quotient (submission / submission18)')
plt.ylabel('Frequency')
plt.title('Histogram of Quotient of views')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('quotient_histogram.png')  # Save the figure
plt.show()