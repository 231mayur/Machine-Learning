from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
advertise=pd.read_csv(r'C:\Users\mayuchav\OneDrive - AMDOCS\Backup Folders\Desktop\ML\advertising.csv')
#sns.heatmap(advertise.isnull())
#sns.distplot(advertise['Daily Time Spent on Site'])
#sns.boxplot(x='Male',y='Daily Time Spent on Site',data=advertise)
advertise.drop(['Timestamp','Ad Topic Line','City','Country'],inplace=True,axis=1)
print(advertise.info())
X=advertise.drop('Clicked on Ad',axis=1)
y=advertise['Clicked on Ad']

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predications=logmodel.predict(x_test)
print(confusion_matrix(y_test,predications))
print(classification_report(y_test,predications))
plt.show()
