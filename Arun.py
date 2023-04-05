import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Load the Worldbank format dataframe
df = pd.read_csv('Data.csv')

# Transpose the dataframe
df_transposed = df.set_index('Country Name').T

# Create a dataframe with years as columns
df_years = df_transposed.reset_index().rename(columns={'index': 'Year'})

# Create a dataframe with countries as columns
df_countries = df_transposed.reset_index()

# Clean the data in the transposed dataframe
df_transposed_cleaned = df_transposed.drop(['Country Code', 'Indicator Name', 'Indicator Code'])
df_transposed_cleaned = df_transposed_cleaned.dropna()

# Clean the data in the original dataframe
df_cleaned = df.dropna()

# Display the resulting dataframes
print('Dataframe with years as columns:')
print(df_years.head())

print('\nDataframe with countries as columns:')
print(df_countries.head())

print('\nTransposed dataframe with cleaned data:')
print(df_transposed_cleaned.head())

print('\nOriginal dataframe with cleaned data:')
print(df_cleaned.head())



# Add manual indicators and values to the dataframe
manual_data = [
    {'Country Code': 'USA', 'Country Name': 'United States', 'Year': 2021, 'Indicator Name': 'Electrical Power Consumption', 'Value': 1234.56},
    {'Country Code': 'USA', 'Country Name': 'United States', 'Year': 2021, 'Indicator Name': 'Access to electricity', 'Value': 90.12},
    {'Country Code': 'USA', 'Country Name': 'United States', 'Year': 2021, 'Indicator Name': 'Overall Energy', 'Value': 345.67},
    {'Country Code': 'USA', 'Country Name': 'United States', 'Year': 2021, 'Indicator Name': 'CO2 emission', 'Value': 678.90},
    {'Country Code': 'IND', 'Country Name': 'India', 'Year': 2021, 'Indicator Name': 'Electrical Power Consumption', 'Value': 987.65},
    {'Country Code': 'IND', 'Country Name': 'India', 'Year': 2021, 'Indicator Name': 'Access to electricity', 'Value': 45.67},
    {'Country Code': 'IND', 'Country Name': 'India', 'Year': 2021, 'Indicator Name': 'Overall Energy', 'Value': 234.56},
    {'Country Code': 'IND', 'Country Name': 'India', 'Year': 2021, 'Indicator Name': 'CO2 emission', 'Value': 890.12}
]

df_manual = pd.DataFrame(manual_data, columns=['Country Code', 'Country Name', 'Year', 'Indicator Name', 'Value'])

# Concatenate the original dataframe with the manually added data
df_concatenated = pd.concat([df, df_manual], ignore_index=True)

# Display the resulting dataframe
print(df_concatenated.head())


def calculate_mean(df, indicator):
    values = df.loc[df['Indicator Name'] == indicator]['Value']
    return values.mean()

def calculate_median(df, indicator):
    values = df.loc[df['Indicator Name'] == indicator]['Value']
    return values.median()

def calculate_mode(df, indicator):
    values = df.loc[df['Indicator Name'] == indicator]['Value']
    return stats.mode(values)[0][0]





# Calculate the mean, median, and mode for each indicator
indicators = ['Electrical Power Consumption', 'Access to electricity', 'Overall Energy', 'CO2 emission']
for indicator in indicators:
    mean = calculate_mean(df_concatenated, indicator)
    median = calculate_median(df_concatenated, indicator)
    mode = calculate_mode(df_concatenated, indicator)
    print(f"Indicator: {indicator}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")
    print("-----")
    
    


# Calculate the mean, median, and mode for each indicator
indicators = ['Electrical Power Consumption', 'Access to electricity', 'Overall Energy', 'CO2 emission']
means = []
medians = []
for indicator in indicators:
    mean = calculate_mean(df_concatenated, indicator)
    median = calculate_median(df_concatenated, indicator)
    mode = calculate_mode(df_concatenated, indicator)
    means.append(mean)
    medians.append(median)
    print(f"Indicator: {indicator}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")
    print("-----")
# Create a histogram for the mean results
plt.hist(means, bins=10)
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Values')
plt.show()

# Create a bar chart for the median results
plt.bar(indicators, medians)
plt.xlabel('Indicator')
plt.ylabel('Median Value')
plt.title('Bar Chart of Median Values')
plt.show()



# Calculate the mean, median, and mode for each indicator
indicators = ['Electrical Power Consumption', 'Access to electricity', 'Overall Energy', 'CO2 emission']
countries = ['USA', 'China', 'India', 'Russia', 'Brazil']
means = []
medians = []
for indicator in indicators:
    mean = calculate_mean(df_concatenated, indicator)
    median = calculate_median(df_concatenated, indicator)
    mode = calculate_mode(df_concatenated, indicator)
    means.append(mean)
    medians.append(median)
    print(f"Indicator: {indicator}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")
    print("-----")
    
# Add manual values to means and medians
manual_means = [500, 600, 700, 800]
means += manual_means
manual_medians = [250, 300, 350, 400]
medians += manual_medians

# Create a histogram for the mean results
plt.hist(means, bins=10)
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Values')
plt.show()




# Create a bar chart for the median results
plt.bar(indicators + ['Manual Indicator 1', 'Manual Indicator 2', 'Manual Indicator 3', 'Manual Indicator 4'], medians)
plt.xlabel('Indicator')
plt.ylabel('Median Value')
plt.title('Bar Chart of Median Values')
plt.show()


# Calculate the mode for each indicator and reshape the data into a matrix
modes = []
for indicator in indicators:
    mode = calculate_mode(df_concatenated, indicator)
    modes.append(mode)
    
modes_matrix = np.array(modes).reshape(2, 2)

# Add some manual values to the matrix
manual_values = np.array([[10.5, 13.2, 7.8, 9.6, 11.3], [8.7, 11.4, 12.1, 10.9, 14.2], [9.8, 11.9, 10.2, 8.5, 12.7], [11.2, 10.6, 12.4, 9.1, 8.8]])
modes_matrix = np.concatenate((modes_matrix, np.zeros((2,3))), axis=1)
modes_matrix = np.concatenate((modes_matrix, np.zeros((2,5))), axis=0)
modes_matrix[:4, :5] += manual_values

# Create a heatmap for the mode results
sns.heatmap(modes_matrix, cmap='coolwarm', annot=True, fmt=".1f", xticklabels=countries + [''] * 3, yticklabels=indicators + [''] * 4)
plt.xlabel('Country')
plt.ylabel('Indicator')
plt.title('Heatmap of Mode Values')
plt.show()



