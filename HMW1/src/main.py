# Contents of /linear-regression-project/linear-regression-project/src/main.py

import pandas as pd
from model.model_LR import LR
from sklearn.metrics import r2_score, mean_squared_error
from visulization.visu import bar_plot

def main():
    

    data = pd.read_csv('data/sampregdata.csv') 

    X = data[['x1']]
    y = data['y']


    model_X1 = LR()


    model_X1.fit(X, y)


    predictions_1 = model_X1.predict(X)


    print("Predictions 1 X:", predictions_1)


    data = pd.read_csv('data/sampregdata.csv') 


    X = data[['x1','x2']]
    y = data['y']


    model_X2 = LR()


    model_X2.fit(X, y)


    predictions_2 = model_X2.predict(X)


    print("Predictions 2 X:", predictions_2)


    mse_1 = mean_squared_error(y, predictions_1)
    mse_2 = mean_squared_error(y, predictions_2)
    print("Mean Squared Error for 1 X:", mse_1)
    print("Mean Squared Error for 2 X:", mse_2)

    bar_plot(mse_1, mse_2)



if __name__ == "__main__":
    main()