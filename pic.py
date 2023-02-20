import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def data2img(folder_path):
    tmp_length = 17
    # Create an empty list to store the dataframes
    data_img = []
    label = []
    # Loop through each file in the folder
    for f in folder_path:
        for file in os.listdir(f):
            if file.endswith(".xlsx"):
                # Read the Excel file and append the dataframe to the list
                file_path = os.path.join(f, file)
                df = pd.read_excel(file_path)
                colunm_names = df.columns
                df = df[colunm_names[3:]]
                x_value = np.array(df)
                scaler = StandardScaler()
                x_value = scaler.fit_transform(x_value)
                #print(np.shape(x_value))
                #x_value = np.reshape(x_value,(len(df),17))
                #print(x_value)
                for i in range(len(df)//17):
                    img = x_value[i:i+17,:]
                    if f == r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/DiabGr":
                        label.append(np.array([0]))
                    else: label.append(np.array([1]))
                    data_img.append(img)
                    img = np.array(data_img, dtype = np.float32)
        #data_img = np.array(data_img)
    return img, label          

if __name__  == '__main__':
    folder_path = [r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/ContrGr", r"/Users/evan/Desktop/studies & research/final year project/yxy/S1Datasets/DiabGr"]
    img, label = data2img(folder_path)
    np.save("./data/train.npy",img)  # 保存文件
    np.save("./data/label.npy",label)  # 保存文件
    print(np.shape(img))
    print(img.dtype)

            


