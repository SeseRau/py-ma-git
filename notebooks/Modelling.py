import pandas as pd
import sklearn as sk
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from autosklearn.regression import AutoSklearnRegressor
import autosklearn.metrics

from pathlib import Path

noise = 0
folder = "/home/sebastian/Dokumente/Python-Git/py-ma-git/workdir/AIS-ModelFrames/"
#filename = f"aisdk-2023-11-08-xs_2_kkn"
#filename = "aisdk-2023-11-08_2_kkn"
filename = "aisdk-2023-11-08-6xs_2_kkn"

#filename = "aisdk-2023-11-08_2_kkn"

for known in range(3, 13, 3):

    train = pd.read_csv(folder+filename+"_train_"+str(known)+"_0.csv")
    test = pd.read_csv(folder+filename+"_test_"+str(known)+"_0.csv")

    train_y = train.loc[:,["output_x","output_y"]]
    test_y = test.loc[:,["output_x","output_y"]]
    train_x = train.drop(columns=["output_x","output_y"])
    test_x = test.drop(columns=["output_x","output_y"])

    test = []
    train = []

    df_res = test_y.copy()

    df_res.rename(columns={"output_x": "known_"+str(known)+"_0", "output_y": "noise_"+str(known)+"_0"}, inplace=True)

    df_res[f"known_"+str(known)+"_0"] = known
    df_res[f"noise_"+str(known)+"_0"] = noise

    # linear regression

    path = Path(folder+filename+"_lin_"+str(known)+"_0.csv")
    if not path.exists():  

        linearpipe = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
        linearpipe.fit(train_x, train_y)
        pickle.dump(linearpipe, open(path, 'wb'))
        linearpipe = []


    # neural network

    path = Path(folder+filename+"_mlp_"+str(known)+"_0..csv")
    if not path.exists(): 

        mlppipe = Pipeline([("scaler", StandardScaler()), ("regressor", MLPRegressor((50)))])
        mlppipe.fit(train_x, train_y)
        pickle.dump(mlppipe, open(path, 'wb'))
        mlppipe = []

    # autosklearn

    path = Path(folder+filename+"_autosk_"+str(known)+"_0.csv")
    if not path.exists(): 

    
        automl = AutoSklearnRegressor(
            #time_left_for_this_task=3600, memory_limit=110*1024, n_jobs=-1, metric=autosklearn.metrics.mean_squared_error
            time_left_for_this_task=30, memory_limit=110*1024, n_jobs=-1, metric=autosklearn.metrics.mean_squared_error
            )
        automl.fit(train_x, train_y, dataset_name="known"+str(known))

        ensemble_dict = automl.show_models()
        print(ensemble_dict)
        pickle.dump(automl, open(path, 'wb'))
        automl = []

    print("Known: "+str(known)+", Noise: "+str(noise))   


