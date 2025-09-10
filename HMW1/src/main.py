# Contents of /linear-regression-project/linear-regression-project/src/main.py

import pandas as pd
from model.model_LR import LR
from sklearn.metrics import r2_score, mean_squared_error

def main():
    

    data = pd.read_csv('data/sampregdata.csv') 

    X = data[['x1']]
    y = data['y']


    model_X1 = LR()


    model_X1.fit(X, y)


    predictions_1 = model_X1.predict(X)


    print("Predictions 1 X:", predictions_1)



if __name__ == "__main__":
    main()