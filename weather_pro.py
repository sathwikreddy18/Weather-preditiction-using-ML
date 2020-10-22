import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import tkinter
from tkinter import *
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#window code
m=Tk()
m.geometry("250x250")
m.title('Weather Predictor')
m.resizable(False,False)

l1=Label(m,text='Min Temp.')
l1.place(x=30,y=80)

l2=Label(m,text='Max Temp.')
l2.place(x=30,y=110)

l3=Label(m,text='')
l3.place(x=80,y=160)


c1s=IntVar()
c1=Entry(m,textvariable=c1s)
c1.place(x=100,y=80)

c2s=IntVar()
c2=Entry(m,textvariable=c2s)
c2.place(x=100,y=110)



df=pd.read_csv("F:\weatherAUS.csv")
#print(df.head())

df.drop(["Sunshine","Evaporation","Cloud3pm","Cloud9am","Date","RISK_MM",'WindGustDir', 'WindDir3pm', 'WindDir9am'],axis=1,inplace=True)
df.dropna(how="any")
df.RainToday.replace({"No":0,"Yes":1},inplace=True)
df.RainTomorrow.replace({"No":0,"Yes":1},inplace=True)
df.drop(["Location"],axis=1,inplace=True)
df.dropna(inplace=True)
train_df=df.drop(["RainTomorrow"],axis=1)
test_df=df[["RainTomorrow"]]
obj=StandardScaler()
new_train_df=obj.fit_transform(train_df)
x_train,y_train,x_test,y_test=train_test_split(new_train_df,test_df,test_size=0.2,random_state=23)
model=LogisticRegression(random_state=0,solver="lbfgs")
model.fit(x_train,x_test)
z=model.predict(y_train)
#teser=classification_report(y_test,z,labels=[1])
s=df.iloc[:5,df.columns!="RainTomorrow"]
t=test_df.head(5)
def test():
    s=[int(c1.get()),int(c2.get()),7.0,50,17,11,22,40,1000,1000,16,9,0]
    s=np.array(s)
    #print(s)
    pred=model.predict(s.reshape(1,-1))
    pred=format(int(round(pred[0],0)),',')
    if pred==0:
        l3['text']="NO RAINFALL"
        return 0
    else:
        l3['text']="THERE WILL BE RAINFALL"
        return 0
b=Button(m,text='Check',command=test)
b.place(x=130,y=190)
m.mainloop()


