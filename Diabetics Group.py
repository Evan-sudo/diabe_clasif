'''
author:Clover
date:2023.02.17
'''

import pandas as pd
import numpy as np
import os
import seaborn as sbn
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

# Set the folder path where the Excel files are stored
folder_path = [r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/ContrGr", \
  r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/DiabGr" ]


# Create an empty list to store the dataframes
dfs = []
# Loop through each file in the folder
for f in folder_path:
    for file in os.listdir(f):
        if file.endswith(".xlsx"):
            # Read the Excel file and append the dataframe to the list
            file_path = os.path.join(f, file)
            df = pd.read_excel(file_path)
            dfs.append(df)

# Concatenate the dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)
print(len(combined_df))
colunm_names = combined_df.columns
dia_data = combined_df[colunm_names[3:]]
new_column = pd.Series(np.append(np.zeros(215678),np.ones(57911)),name = 'label')
dia_data = pd.concat([dia_data,new_column], axis = 1)
print(dia_data)
corr = dia_data.corr()
print(corr)
mpl.use('TkAgg')  # !IMPORTANT
sbn.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

## data preprocessing
X = dia_data.drop('label',axis = 1).values
Y = dia_data['label'].values
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
print(X_scaler)

'''
## PCA
pca = PCA(n_components=8)
newX = pca.fit_transform(X_scaler)
print(pca.explained_variance_ratio_)
print(newX)
'''
