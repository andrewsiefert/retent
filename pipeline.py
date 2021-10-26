from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process.kernels import RBF
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import matplotlib.pyplot as plt
import shap

from transformers import *


class RetentModel:

    def __init__(self):
        self.prepared_data = None
        self.prepared_df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cat_cols = None
        self.num_cols = None
        self.bin_cols = None
        self.preprocessor = None
        self.features = None
        self.X_train_prepared = None
        self.X_test_prepared = None
        self.X_train_prepared_df = None
        self.X_test_prepared_df = None
        self.model = None
        self.pipeline = None
        self.tune_best = None
        self.tune_results = None
        self.platt = None
        self.iso = None
        self.gp = None
        self.explainer = None

    def get_xy(self, data, response) -> None:

        # get predictors (X) and response (y)
        self.response = response

        self.X = data.drop([response], axis=1)

        self.y = data[response]

    def get_column_types(self):
        self.cat_cols = self.X.columns[self.X.dtypes == 'object']
        self.num_cols = self.X.columns[(self.X.dtypes != 'object') & (self.X.apply(lambda x: x.nunique()) > 2)]
        self.bin_cols = self.X.columns[(self.X.dtypes != 'object') & (self.X.apply(lambda x: x.nunique()) == 2)]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def create_pipeline(self, data, response):

        self.get_xy(data, response)
        self.get_column_types()
        self.split_data()

        # categorical transformer
        lumper = Lumper(min_cases=200)
        onehot = OneHotEncoder2(sparse=False, handle_unknown='ignore')
        imputer = SimpleImputer2(strategy="constant", fill_value=-999)
        cat_pipe = Pipeline2([('lumper', lumper), ('onehot', onehot), ('imputer', imputer)])

        # numeric transformer
        num_imputer = SimpleImputer2(strategy="constant", fill_value=-999999)

        # create transformer
        self.preprocessor = ColumnTransformer([('num', num_imputer, self.num_cols),
                                              ('cat', cat_pipe, self.cat_cols),
                                              ('bin', num_imputer, self.bin_cols)])
        # create xgboost
        if self.y_train.nunique() == 2:
            self.classifier = xgb.XGBClassifier()
        else:
            self.classifier = xgb.XGBRegressor()

        self.pipeline = Pipeline([('preproc', self.preprocessor), ('xgboost', self.classifier)])
        self.model = self.pipeline.named_steps['xgboost']


    def fit(self, data=None, response=None):

        if self.pipeline == None:
            self.create_pipeline(data, response)

        if self.tune_best == None:
            print("Fitting using default hyperparameters")
            self.pipeline.fit(self.X_train, self.y_train)
        else :
            self.pipeline.set_params(**self.tune_best)
            self.pipeline.fit(self.X_train, self.y_train)

        self.X_train_prepared = self.preprocessor.transform(self.X_train)
        self.X_test_prepared = self.preprocessor.transform(self.X_test)

        self.features = [re.sub("^.+__", "", x) for x in self.preprocessor.get_feature_names()]

        self.X_train_prepared_df = pd.DataFrame(self.X_train_prepared, columns=self.features)
        self.X_test_prepared_df = pd.DataFrame(self.X_test_prepared, columns=self.features)

    def process_data(self, X, df=False):
        X_trans = self.preprocessor.transform(X)
        if df:
            return pd.DataFrame(X_trans, columns = self.features)
        else:
            return X_trans

    def tune(self, n = 20):

        pipeline = self.pipeline

        # define optimization function
        def objective(space):
            params = {key: int(value) if value % 1 == 0 else value for key, value in space.items()}

            print("Training with params: ")
            for key, value in params.items():
                if type(value) == float:
                    print(key + ": " + str(round(value, 3)))
                else:
                    print(key + ": " + str(value))

            # create pipeline
            pipeline.set_params(**params)

            cv = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring='roc_auc')
            auc = np.mean(cv)
            score = 1 - auc

            print("\n" + "AUC: ")
            print(str(round(auc, 3)) + "\n")

            params['auc'] = auc

            param_list.append(params)

            return score

        # define search space
        space = {'preproc__cat__lumper__min_cases': hp.quniform('preproc__cat__lumper__min_cases', 10, 500, 1),
                 'xgboost__max_depth': hp.quniform('xgboost__max_depth', 3, 10, 1),
                 'xgboost__subsample': hp.uniform('xgboost__subsample', 0.3, 1.0),
                 'xgboost__colsample_bytree': hp.uniform('xgboost__colsample_bytree', 0.3, 1.0),
                 'xgboost__gamma': hp.uniform('xgboost__gamma', 0.0, 0.5),
                 'xgboost__reg_alpha': hp.loguniform('xgboost__reg_alpha', -10, 0),
                 'xgboost__reg_lambda': hp.loguniform('xgboost__reg_lambda', -10, 3),
                 'xgboost__min_child_weight': hp.uniform('xgboost__min_child_weight', 0.1, 10),
                 'xgboost__n_estimators': hp.quniform('xgboost__n_estimators', 50, 200, 1)
                 }

        # run hyperparameter search
        param_list = []

        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=n)

        self.tune_best = {key: int(value) if value % 1 == 0 else value for key, value in best.items()}
        self.tune_results = pd.DataFrame(param_list).sort_values('auc', ascending=False)

    def tune_fit(self, data=None, response=None, n=20):

        if self.pipeline == None:
            self.create_pipeline(data, response)

        self.tune(n=n)
        self.fit()

    def predict(self, X):
        return self.pipeline.predict_proba(X)[:,1]

    def evaluate(self):
        return roc_auc_score(self.y_test, self.predict(self.X_test))

    def calibrate(self):
        pred_nocalib = self.predict(self.X_test).reshape(-1, 1)

        # Platt scaling
        self.platt = LogisticRegression(penalty="none", solver='sag')
        self.platt.fit(pred_nocalib, self.y_test)

        # isotonic regression
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(X=pred_nocalib.reshape(-1).astype('float'), y=self.y_test)

        # Gaussian process
        #self.gp = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        #self.gp.fit(X=pred_nocalib, y=self.y_test)

    def predict_calib(self, X, calibrator):

        pred_nocalib = self.predict(X)

        if calibrator=='platt':
            return self.platt.predict_proba(pred_nocalib.reshape(-1, 1))[:, 1]

        if calibrator=='iso':
            return self.iso.predict(pred_nocalib.reshape(-1))

        if calibrator=='gp':
            return self.gp.predict_proba(pred_nocalib.reshape(-1, 1))[:, 1]

    def calib_plot(self):

        X = self.X_test
        preds = pd.DataFrame({'actual': self.y_test,
                              'no_calib': self.predict(X),
                              'platt': self.predict_calib(X, 'platt').reshape(-1),
                              'iso': self.predict_calib(X, 'iso').reshape(-1)
                              })

        def calibration_plot(y_obs, y_pred):
            calib_bins = pd.DataFrame(np.column_stack(calibration_curve(y_obs, y_pred)),
                                      columns=['prob_true', 'prob_pred'])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(calib_bins['prob_pred'], calib_bins['prob_true'], linewidth=2)
            ax.plot([0, 1], [0, 1], '--', color="gray")
            plt.draw()

        for c in preds.columns[1:]:
            calibration_plot(self.y_test, preds[c])
            plt.title(c)


    def get_explainer(self):

        booster = self.model.get_booster()
        model_bytearray = booster.save_raw()[4:]

        def myfun(self=None):
            return model_bytearray

        booster.save_raw = myfun

        self.explainer = shap.TreeExplainer(booster)


    def get_shap_values(self, X):
        shap_values = self.explainer.shap_values(self.process_data(X))
        return pd.DataFrame(shap_values, columns=self.features)

    def shap_plot(self, X):
        shap_values = self.explainer.shap_values(self.process_data(X))
        # plot SHAP values
        shap.summary_plot(shap_values, self.process_data(X),
                          feature_names=self.features)