# Machine learning pipeline for retention prediction

This repository contains code to demo a Python machine learning pipeline I built to predict and explain student retention and other outcomes when I was working at Ithaca College. In the real world, the model was deployed in the cloud (first AWS, then Azure) and ran daily to pull in student data from various sources, generate new predictions and explanations, and update a Tableau dashboard that displayed information to campus stakeholders. I can't share the full project or real student data due to privacy concerns, but this repo contains the core parts of the prediction pipeline. This includes: 

- Custom scikit-learn feature transformers to preprocess the data
- XGBoost classifier with Bayesian hyperparameter tuning via [hyperopt](https://github.com/hyperopt/hyperopt) to generate predictions
- Explanations of model predictions using [SHAP](https://www.example.com), a game-theoretic approach

This is all implemented in an object-oriented programming framework in which an entire pipeline is stored as an object of the RetentPipe class. I originally built the pipeline to predict student retention but it could be used to predict any binary outcome from any combination of continuous and categorical features. You can see a demo of the pipeline using tree survival data by running `survival_demo.py`.
