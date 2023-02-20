import pandas as pd
import os
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np


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


## data preprocessing
X = dia_data.drop('label',axis = 1).values
label1 = dia_data['label'].values
scaler = StandardScaler()
train1 = scaler.fit_transform(X)

print(np.shape(train1))
print(np.shape(label1))
print(label1)

np.save("./data/train_non.npy",train1)  
np.save("./data/label_non.npy",label1) 