# Import torch and pandas (assuming it's not already imported)
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the full path to the CSV file
file_path = r"XXX\Online Retail.csv" # Insert path to dataset

# Load the dataset
df = pd.read_csv(file_path)

# Select only the "Description" column and ensure string type
df_unique = df[["Description"]].astype(str)

# Remove duplicate descriptions (case-sensitive)
df_unique = df_unique.drop_duplicates(subset="Description")

# Handle missing values (empty descriptions)
df_unique = df_unique.dropna(subset=["Description"])

# Load the ALBERT model and tokenizer (change 'albert-base-v2' if desired)
model_name = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels if needed

# Define candidate labels
candidate_labels = ["utilitarian", "decorative"]


# Function to classify a product description
def classify_product(description):
    # Tokenize the description
    inputs = tokenizer(description, return_tensors="pt")

    # Perform classification with the model
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Softmax for probabilities

    # Get the label with the highest probability
    predicted_label = candidate_labels[torch.argmax(predictions, dim=-1).item()]
    return predicted_label


# Apply classification to data
df_unique['classification'] = df_unique['Description'].apply(classify_product)

# Display the updated dataframe with classifications
print(df.head())

# Merge df with df_unique based on "Description" (optional)
df = df.merge(df_unique[['Description', 'class']], how='left', on='Description')
