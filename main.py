from Perceptron import binary_classification
import numpy as np
import pandas as pd

df = pd.read_csv('dataset/Raisin_Dataset.csv')
df['Class'] = df['Class'].apply(lambda x: 1 if x == 'Kecimen' else 0)
df_col = df['Class']
df_test = df.loc[0:350]
expected_test = df_test['Class']
df_test = df_test.drop(columns = ['Class'])
df = df.drop(columns=['Class'])

print(df_test.head())
print(df_test.shape)
if __name__ == "__main__":
    result = binary_classification(df_test, expected_test, df, df_col, epochs=10)
    print(result)
