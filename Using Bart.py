import pandas as pd
from transformers import pipeline

# Specify the full path to your CSV file
file_path = r"C:\Users\oscar\Documents\Projects\Online Retail Project\online+retail\Online Retail.csv"

# Load your dataset
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Select only the "Description" column and ensure string type
df_unique = df[["Description"]].astype(str)

# Remove duplicate descriptions (case-sensitive)
df_unique = df_unique.drop_duplicates(subset="Description")

# Print the modified DataFrame (optional)
print(df)

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["utilitarian", "decorative"]


# Function to classify a product description
def classify_product(description):
    result = classifier(description, candidate_labels)
    return result['labels'][0]


# Example usage with the dataset
df_unique['classification'] = df_unique['Description'].apply(classify_product)

# Display the updated dataframe with classifications
print(df.head())

# Merge df with df_unique based on "Description"
df = df.merge(df_unique[['Description', 'class']], how='left', on='Description')
