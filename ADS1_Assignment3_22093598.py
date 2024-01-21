# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Function to read data from Excel or CSV files
def read_data(file):
    """
    Read data from an Excel or CSV file.

    Parameters:
    - file (str): File path or URL.

    Returns:
    - df_clean (pd.DataFrame): Cleaned DataFrame.
    - df_t (pd.DataFrame): Transposed DataFrame.
    """
    if ".xlsx" in file:
        df = pd.read_excel(file, index_col=0, delimiter=',')
    elif ".csv" in file:
        df = pd.read_csv(file, index_col=0, delimiter=',')
    else:
        print("Invalid filetype. Please provide a valid Excel (.xlsx) or CSV (.csv) file.")
        return None, None

    df_clean = df.dropna(axis=1, how="all").dropna()
    df_t = df_clean.transpose()

    return df_clean, df_t

def perform_clustering(dataframe, features, n_clusters=4):
    """
    Perform KMeans clustering on specified features.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - features (list): List of features for clustering.
    - n_clusters (int): Number of clusters for KMeans. Default is 4.

    Returns:
    - labels (np.ndarray): Cluster labels.
    - centres (np.ndarray): Cluster centres.
    """
    scaler = RobustScaler()
    to_clust = dataframe[features]
    scaler.fit(to_clust)
    norm = scaler.transform(to_clust)

    clusters = KMeans(n_clusters=n_clusters, n_init=20)
    clusters.fit(norm)
    labels = clusters.labels_
    centres = clusters.cluster_centers_
    centres = scaler.inverse_transform(centres)

    return labels, centres

# Define the polynomial function
def poly(x, a, b, c):
    return a * x**2 + b * x + c

GDP_Growth_data, GDP_Growth_Transposed = read_data("https://raw.githubusercontent.com/JawadDS/ADS_Assignment3/main/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_6298243.csv")
Gross_Savings_data, Gross_Savings_Transposed = read_data("https://raw.githubusercontent.com/JawadDS/ADS_Assignment3/main/API_NY.GNS.ICTR.ZS_DS2_en_csv_v2_6298647.csv")

# Extract GDP Growth and Gross Savings for Pakistan
GDP_Growth_Pak = GDP_Growth_Transposed.loc['1990':'2020', 'Pakistan'].copy()
Gross_Savings_Pak = Gross_Savings_Transposed.loc['1990':'2020', 'Pakistan'].copy()

# Merge and set index for the DataFrame
df_Pakistan = pd.merge(GDP_Growth_Pak, Gross_Savings_Pak, on=GDP_Growth_Pak.index, how="outer")
df_Pakistan.rename(columns={'key_0':'Years', 'Pakistan_x': 'GDP Growth', 'Pakistan_y': 'Gross Savings'}, inplace=True)
df_Pakistan = df_Pakistan.set_index('Years')

# Perform clustering
cluster_labels, cluster_centres = perform_clustering(df_Pakistan, ["GDP Growth", "Gross Savings"], n_clusters=4)

# Scatter plot with clustering results
#cm = ['red', 'blue', 'green', 'purple']
plt.scatter(df_Pakistan["GDP Growth"], df_Pakistan["Gross Savings"], 10, cluster_labels, marker="o", cmap='viridis')
plt.scatter(cluster_centres[:, 0], cluster_centres[:, 1], 45, "k", marker="d")
plt.gca().set_facecolor('#F8CBAD')
plt.xlabel("GDP Growth")
plt.ylabel("Gross Savings")
plt.title("GDP Growth vs Gross Savings (Pakistan) with Clustering")
plt.show()

# Prepare DataFrame for fitting
df_Pakistan = df_Pakistan.reset_index()
df_Pakistan["GDP Growth"] = pd.to_numeric(df_Pakistan["GDP Growth"])
df_Pakistan["Years"] = pd.to_numeric(df_Pakistan["Years"])

# Extract features and target for curve fitting
years = df_Pakistan['Years'].values
gdp_growth = df_Pakistan['GDP Growth'].values
gross_savings = df_Pakistan['Gross Savings'].values

# Curve fitting for GDP Growth
params_gdp, _ = curve_fit(poly, years, gdp_growth)

# Curve fitting for Gross Savings
params_savings, _ = curve_fit(poly, years, gross_savings)

# Generate future years for forecasting
future_years = np.arange(1990, 2031)

# Forecast GDP Growth and Gross Savings
forecast_gdp = poly(future_years, *params_gdp)
forecast_savings = poly(future_years, *params_savings)

# Plotting
plt.figure(figsize=(12, 6))

# Plot GDP Growth
plt.subplot(1, 2, 1)
plt.scatter(df_Pakistan['Years'], df_Pakistan['GDP Growth'], label='Actual Data (GDP Growth)', c='blue')
plt.plot(future_years, forecast_gdp, label='Forecasted GDP Growth', c='red')
plt.gca().set_facecolor('#F8CBAD')
plt.xlabel('Years')
plt.ylabel('GDP Growth')
plt.title('GDP Growth Forecast for Pakistan')
plt.legend()

# Plot Gross Savings
plt.subplot(1, 2, 2)
plt.scatter(df_Pakistan['Years'], df_Pakistan['Gross Savings'], label='Actual Data (Gross Savings)', c='green')
plt.plot(future_years, forecast_savings, label='Forecasted Gross Savings', c='orange')
plt.gca().set_facecolor('#F8CBAD')
plt.xlabel('Years')
plt.ylabel('Gross Savings')
plt.title('Gross Savings Forecast for Pakistan')
plt.legend()

plt.tight_layout()
plt.show()

# Additional line plot for both GDP Growth and Gross Savings
plt.figure(figsize=(12, 6))

# Line plot for GDP Growth
plt.plot(df_Pakistan['Years'], df_Pakistan['GDP Growth'], label='GDP Growth', c='blue')
plt.gca().set_facecolor('#F8CBAD')
plt.xlabel('Years')
plt.ylabel('GDP Growth')
plt.title('GDP Growth and Gross Savings for Pakistan (1990 to 2020)')
plt.legend(loc='upper left')

# Create a second y-axis for Gross Savings on the right
ax2 = plt.gca().twinx()
ax2.plot(df_Pakistan['Years'], df_Pakistan['Gross Savings'], label='Gross Savings', c='green')
ax2.set_ylabel('Gross Savings')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
