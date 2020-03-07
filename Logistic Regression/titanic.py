from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv(r'C:\Users\mayuchav\OneDrive - AMDOCS\Backup Folders\Desktop\ML\train.csv')
#print(titanic.head())

def imput(cols): #this function will replace all colunm with specfied values
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

titanic['Age'] = titanic[['Age', 'Pclass']].apply(imput,axis=1) # this will call the imput methos and apply on every single row
print(titanic['Age'])
sns.heatmap(titanic.isnull()) #To get the null values on our dataset
'''drop coloumn which are not useful'''
titanic.drop('Cabin',axis=1,inplace=True)
'''machine learning algos are not work on string values to get appropriate int value need to use get_dummies() method  if the
data in catagorical form
'''
sex=pd.get_dummies(titanic['Sex'],drop_first=True)
em=pd.get_dummies(titanic['Embarked'],drop_first=True)
titanic=pd.concat([titanic,sex,em],axis=1)
titanic.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
print(titanic.head())
'''Now let's train our model'''
x=titanic.drop('Survived',axis=1)
y=titanic['Survived']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3)
logistic=LogisticRegression()
logistic.fit(X_train,Y_train)
predictions=logistic.predict(X_test)
classification_report(Y_test,predictions)
confusion_matrix(Y_test,predictions)
plt.show()
