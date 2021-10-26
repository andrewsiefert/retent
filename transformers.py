from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import re 


# Passthrough (no transformation)
class Passthrough(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        self.features = X.columns
        return self
        
    def transform(self, X):
        return X
    
    def get_feature_names(self):
        return self.features


# Create missing value dummy variables
class MissingIndicator2(MissingIndicator):

    def fit(self, X, y = None):
        super().fit(X, y)
        missing_ind = self.features_.tolist()
        missing_cols = X.columns[missing_ind].tolist()
        self._missing_colnames = [x + '_missing' for x in missing_cols]
        return self
        
    def transform(self, X):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self._missing_colnames)

    def get_feature_names(self):
        return self._missing_colnames 
    
# SimpleImputer that includes feature names
class SimpleImputer2(SimpleImputer):
        
    def fit(self, X, y = None):
        super().fit(X, y)
        self._features = X.columns.tolist()
        return self

    def transform(self, X, y = None):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self._features)

    def get_feature_names(self):
        return self._features     
     
# IterativeImputer with feature names
class IterativeImputer2(IterativeImputer):
        
    def fit_transform(self, X, y = None):
        out = super().fit_transform(X, y)
        self._features = X.columns.tolist()
        return pd.DataFrame(out, columns = self._features)
    
    def transform(self, X):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self._features)

    def get_feature_names(self):
        return self._features  
    
# PowerTransformer with feature names
class PowerTransformer2(PowerTransformer):
        
    def fit(self, X, y = None):
        super().fit(X, y)
        self._features = X.columns.tolist()
        return self

    def transform(self, X, y = None):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self._features)

    def get_feature_names(self):
        return self._features   
    
# Bin continuous variable and get feature names
class Binner(KBinsDiscretizer):
   
    def fit(self, X, y=None):
        super().fit(X, y)
        edges = self.bin_edges_
        
        bin_names = []

        for i in range(0, edges.shape[0]):
            bins = [X.columns[i] + '_' + x for x in list(map(lambda x: str(round(x, 1)), edges[i]))] 
            bins.pop(0)
            bin_names.append(bins)

        flatten = lambda l: [item for sublist in l for item in sublist]

        self._features =  flatten(bin_names)
        return self
                
    def transform(self, X, y = None):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self._features) 
    
    def get_feature_names(self):
        return self._features   

# lump rare levels of categorical variable into "rare" category
class Lumper(BaseEstimator, TransformerMixin):
    
    def __init__(self, min_cases = 20):
        self.min_cases = min_cases
        self.features = None
        
    def fit(self, X, y = None):
        
        rare_cols = []
        rare_levels = []
        
        for col in X:        
            rare_check = X[col].value_counts() < self.min_cases
            rare_levs = list(rare_check.index[rare_check]) 
            if len(rare_levs) > 0:
                rare_cols.append(col)
                rare_levels.append(rare_levs)
        
        self.features = X.columns
        self.rare_cols = rare_cols
        self.rare_levels = rare_levels
        return self
    
    def transform(self, X, y = None):
        
        out = X.copy()
        out_rare = out[self.rare_cols]
        for i in range(len(self.rare_cols)):
            out_rare.iloc[:,i] = out_rare.iloc[:,i].replace(self.rare_levels[i], "other")
        out[self.rare_cols] = out_rare
        return out
    
    def get_feature_names(self):
        return self.features
    
    def get_pars(self):
        return self.rare_cols, self.rare_levels

# one-hot encoder that includes feature names
class OneHotEncoder2(OneHotEncoder):
    
    def fit(self, X, y=None):
        super().fit(X)
        
        # get feature names
        raw_names = super().get_feature_names()
        level_names = pd.DataFrame({'raw_names':raw_names})
        level_names = level_names.raw_names.str.split("_", n = 1, expand = True)
        level_names.columns = ['feature', 'level']
        features = X.columns
        level_n = X.apply(lambda x: len(x.value_counts()))

        #if self.drop == 'first':
         #   level_names = level_names.groupby('feature').apply(lambda x: x.iloc[1:,:]) 
          #  level_n = level_n - 1
            
        cat_names = np.repeat(features, level_n)
        self._feature_names = [a + '_' + b for a, b in zip(cat_names, level_names.level)]

        return self
    
    def get_feature_names(self):
        return self._feature_names
        
    def transform(self, X, y = None):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self._feature_names)
    

# create sklearn pipeline with feature names
class Pipeline2(Pipeline):
    
    def get_feature_names(self):  
        last_step = list(self.named_steps.keys())[-1]
        self._feature_names = self.named_steps[last_step].get_feature_names()
        return self._feature_names
    
# FeatureUnion with feature names
class FeatureUnion2(FeatureUnion):
    
    def get_feature_names(self):
        feature_names = super().get_feature_names()
        return [re.sub('^.+__', '', x) for x in feature_names]
    
    def transform(self, X, y = None):
        out = super().transform(X)
        return pd.DataFrame(out, columns = self.get_feature_names())
    
    def fit_transform(self, X, y = None):
        out = self.fit(X, y).transform(X)
        return pd.DataFrame(out, columns = self.get_feature_names())


# create custom transformer to replace rare categories with "rare"
def replace_rare(x, n = 20):
    counts = x.value_counts()
    rare = counts.index[counts <= n]
    if len(rare) > 0:
        return x.replace(rare, "rare")
    else:
        return x


# scale continuous variables by group
class GroupScaler(BaseEstimator, TransformerMixin):
    def __init__(self, group):
        self._group = group
        self._features = None
        super().__init__()

    def fit(self, X, y=None):
        self._features = X.columns
        return self

    def transform(self, X, y = None):
        return X.groupby(self._group).apply(lambda x: (x-np.mean(x))/np.std(x))
    
    def get_feature_names(self):
        return self._features


# convert strings to dates
def to_date(x):
    return pd.to_datetime(x.replace(r'^\s*$', np.nan, regex=True))

# clean up strings
def clean_string(x, fill = "missing"): 
    out = x.astype('str').str.strip().fillna(fill)
    return out.replace(r'^\s*$', fill, regex=True)

def clean_numeric(x):
    return clean_string(x, "0").astype('int')

def binarize_numeric(x):
    return np.where(clean_numeric(x) > 0, 1, 0)


# count NaN's
def count_nan(x):
    return x.isna().sum()


### summarize variable by group
    
def group_summary(x, group):
    missing = x.groupby(group).apply(count_nan)
    unique_vals = x.unique().shape[0]
    data_type = x.dtypes
    if (data_type in ['int', 'float']):
        if unique_vals > 10:
            summary =  x.groupby(group).agg(['mean', 'min', 'max'])
        else:
            summary = pd.crosstab(x, group, normalize = 1)
    else:
        summary = pd.crosstab(x, group, normalize = 1)
    return summary, missing

