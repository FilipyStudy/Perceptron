from Perceptron import binary_classification
import numpy as np
import pandas as pd

df = pd.read_csv('dataset/Raisin_Dataset.csv')
df['Class'] = df['Class'].apply(lambda x: 1 if x == 'Kecimen' else 0)
df_col = df['Class']
df = df.drop(columns=['Class'])

if __name__ == "__main__":
    print(df.head())
    print(df.shape)
    weights = binary_classification(df, df_col)
    print(weights)