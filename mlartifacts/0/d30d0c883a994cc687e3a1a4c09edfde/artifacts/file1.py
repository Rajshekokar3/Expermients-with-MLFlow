import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

#load wine dataset 
wine=load_wine()
X=wine.data 
Y=wine.target


#Train test split
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.10,random_state=42)

#Define the params for EF models 
max_depth=8
n_estimators=20


with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,Y_train)
    

    y_pred=rf.predict(x_test)
    accuracy_score=accuracy_score(y_test,y_pred)

    mlflow.log_metric("accuracy",accuracy_score)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)

    #Creating a confusion metrix plot 
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True ,fmt='o', cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion metrix")


    #save plot locally
    plt.savefig("Confusion-metrix.png")

    #log artifcat using mlflow
    mlflow.log_artifact("Confusion-metrix.png")
    mlflow.log_artifact(__file__)


    print(accuracy_score)