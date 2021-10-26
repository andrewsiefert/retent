import pandas as pd
from joblib import dump

import pipeline as rp

# load survival data
data = pd.read_csv("data/survival_data.csv")

# instantiate model
pipe = rp.RetentModel()

# set up pipeline using survival data
pipe.create_pipeline(data, 'surv')

# tune XGBoost classifier
pipe.tune(n=10)

# fit pipeline using tuned hyperparameters
pipe.fit()

# evaluate model AUC on test set
pipe.evaluate()

# calibrate predictions using Platt scaling (logistic regression) and isotonic regression
pipe.calibrate()
pipe.calib_plot()

# explain the model's predictions using SHAP
pipe.get_explainer()
pipe.shap_plot(pipe.X_test)

# save fitted pipeline
dump(pipe, 'models/survival_pipeline.joblib')

