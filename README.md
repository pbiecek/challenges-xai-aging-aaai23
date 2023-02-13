# Challenges facing the explainability of age prediction models: case study for two modalities

by [Mikolaj Spytek](https://github.com/mikolajsp), [Weronika Hryniewska](https://github.com/Hryniewska), Jarosław Żygierewicz, Jacek Rogala, [Przemyslaw Biecek](https://github.com/pbiecek)

**Supplementary materials**

## EEG

### Models

- [Linear Regression Model](./EEG/LinearRegression.obj)
- [Multi Layer Perceptron](./EEG/MultiLayerPerceptron.obj)

### Sample data for inference
- [10 sample EEG Recordings](./EEG/sample_data.csv)

### Code example

```python
import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


data = pd.read_csv("sample_data.csv", index_col=0)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

with open("LinearRegression.obj", "rb") as f:
    linreg = pickle.load(f)

with open("MultiLayerPerceptron.obj", "rb") as f:
    mlp = pickle.load(f)


prediction_linreg = linreg.predict(X)
prediction_mlp = mlp.predict(X)
```



## X-ray

### Model

- [XGBoost Regression Model](./X-ray/model_xgb_regressor.json)
- [Catboost Regression Model](./X-ray/model_catboost_regressor.json)

### Sample data for inference

