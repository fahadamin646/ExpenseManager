import pandas as pd
import pickle
def getPrediction(Cloth,Food,Fuel,Holiday,Home,Kids,Pharm,Shopping,Transport):
    lst=[[Cloth,Food,Fuel,Holiday,Home,Kids,Pharm,Shopping,Transport]]
    df=pd.DataFrame(lst,columns=['Cloth','Food','Fuel','Holiday','Home','Kids','Pharm','Shopping','Transport'])
    with open('stand_scalar', 'rb') as f:
        sc=pickle.load(f)
    with open('model', 'rb') as f:
        ppn = pickle.load(f)
    dataf=sc.transform(df)
    pred=ppn.predict(dataf)
    return str(pred[0])

