import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:\Data Science\Machine Learning\ML Dataset\Medicalpremium.csv")
df.head()

x = df.drop(columns="PremiumPrice")
x.head()

y = df['PremiumPrice']
y.head()

scale = StandardScaler()
x1 = scale.fit_transform(x)
x1

x_train,x_test,y_train,y_test = train_test_split(x1,y,train_size=0.7,random_state=42)
#x_train,x_test,y_train,y_test = train_test_split(x1,y,train_size=0.7)
#print(x_train[0][0])

model = LinearRegression()

model.fit(x_train,y_train)

#print (model.predict(x_test))



