import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print(" --- exibindo a estrutura do dataframe ---")
print(df.head())
time.sleep(1) #espera de 1 segundo para exibir a proxima informação

print(" --- exibindo a distribuição das classes ---")
print(df["Class"].value_counts(normalize=True))
time.sleep(1)

df["Amount_log"] = np.log1p(df["Amount"])

scaler = StandardScaler()

#normalizando a coluna "Amount" e criando uma nova coluna "Amount_scaled"
#para evitar que a escala da coluna "Amount" influencie o modelo, é importante normalizar os valores dessa coluna.

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

#treinando o modelo de regressão logística 
node1 = LogisticRegression(max_iter=1000)

node1.fit(x_train, y_train)

y_pred = node1.predict(x_test)

print(" --- exibindo o relatório de classificação ---")
print(classification_report(y_test, y_pred))
time.sleep(1)