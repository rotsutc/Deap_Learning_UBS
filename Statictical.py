import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Đường dẫn tương đối dựa trên vị trí file .py
current_dir = os.path.dirname(__file__)  # Lấy đường dẫn thư mục chứa file .py
file_path = os.path.join(current_dir, "Dataset.csv")
    
# Đọc file CSV
data = pd.read_csv(file_path)

# # Display the first few rows of the dataset to understand its structure
# data.head(), data.describe(), data.info()



# # Plot histograms and KDE for all features
# for column in data.columns:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(data[column], kde=True, stat="density", bins=30, color="blue", alpha=0.6)
#     plt.title(f"Distribution of {column}")
#     plt.xlabel(column)
#     plt.ylabel("Density")
#     plt.show()
    
#     # Statistical test for normality (Shapiro-Wilk Test)
#     stat, p = shapiro(data[column])
#     print(f"Shapiro-Wilk Test for {column}: Stat={stat:.4f}, P-value={p:.4e}")
#     if p > 0.05:
#         print(f"{column} likely follows a normal distribution.")
#     else:
#         print(f"{column} does NOT follow a normal distribution.")


# Function to fit and test distributions
def test_distribution(data, column_name):
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(data)
    
    # Kolmogorov-Smirnov Test for normal distribution
    ks_stat, ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    
    # Display results
    print(f"Testing for {column_name}:")
    print(f"- Shapiro-Wilk Test: Stat={shapiro_stat:.4f}, P-value={shapiro_p:.4e}")
    print(f"- Kolmogorov-Smirnov Test (Normal): Stat={ks_stat:.4f}, P-value={ks_p:.4e}")
    
    # Interpretation
    if shapiro_p > 0.05:
        print(f"  {column_name}: Likely follows a normal distribution (Shapiro-Wilk).")
    else:
        print(f"  {column_name}: Does NOT follow a normal distribution (Shapiro-Wilk).")
        
    if ks_p > 0.05:
        print(f"  {column_name}: Fits normal distribution (Kolmogorov-Smirnov).")
    else:
        print(f"  {column_name}: Does NOT fit normal distribution (Kolmogorov-Smirnov).")

    print("-" * 50)

# Applying the tests to all columns
for column in data.columns:
    test_distribution(data[column], column)

# Visualize Kernel Density Estimation (KDE) for one feature as an example
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.kdeplot(data['X1'], label='X1', fill=True, color='blue', alpha=0.5)
sns.kdeplot(data['X9'], label='X9', fill=True, color='green', alpha=0.5)
sns.kdeplot(data['Output'], label='Output', fill=True, color='red', alpha=0.5)
plt.title("Kernel Density Estimation (KDE)")
plt.legend()
plt.show()