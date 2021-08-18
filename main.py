import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

dataset=pd.read_csv("Students_Score.csv")



dataset.head()
dataset.describe()

dataset.plot(x='Hours',y='Scores',style="*")
plt.title('Student mark prediction')
plt.xlabel('Hours')
plt.ylabel('Percentage marks')
plt.show()

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.intercept_)

print(regressor.coef_)

y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
df