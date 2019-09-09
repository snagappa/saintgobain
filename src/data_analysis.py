#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:35:18 2019

@author: snagappa
"""
import io
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

import data_loader


RANDOM_STATE = 3141592654
_EPS = np.finfo(np.float).eps


df = data_loader.df
df_referendum = data_loader.df_referendum
df_latlon = data_loader.df_latlon
df_pop_str_edu = data_loader.df_pop_str_edu
df_popfiscal = data_loader.df_popfiscal
dfcorr = data_loader.dfcorr


class Regressor(object):
    """Build a collection of regressors - 1 per referendum category."""
    def __init__(self, regressor_class, *args, **kwargs):
        self.regressor = [regressor_class(*args, **kwargs) for ix in range(4)]

    def fit(self, X, y):
        """Call fit method for the regressor for each referendum category"""
        X = np.asarray(X)
        y = np.asarray(y)
        for colix in range(4):
            self.regressor[colix].fit(X, y[:, colix])

    def predict(self, X):
        """Predict the referendum outcome for each category."""
        ypred = [regressor_.predict(X) for regressor_ in self.regressor]
        ypred = np.vstack(ypred).T
        ypred[ypred > 100] = 100
        ypred[ypred < 0] = 0
        #ypred /= (ypred.sum(axis=1)[:, np.newaxis] + _EPS)
        return ypred


def _mae(regressor, x, ytrue):
    """Return the Mean Absolute Error for each referendum category"""
    ypred = regressor.predict(x)
    mae = mean_absolute_error(
        ytrue, ypred, multioutput="raw_values")
    return mae


def _rmse(regressor, x, ytrue):
    """Return the Mean Squared Error for each referendum category"""
    ypred = regressor.predict(x)
    mse = mean_squared_error(
        ytrue, ypred, multioutput="raw_values")
    rmse = mse**0.5
    return rmse


def _r2(regressor, x, ytrue):
    """Return the R2 score for each referendum category"""
    ypred = regressor.predict(x)
    r2 = r2_score(
        ytrue, ypred, multioutput="raw_values")
    return r2


def _build_regressor(data, regressor_class, *args, **kwargs):
    """Train an instance of regressor_class and compute RMSE."""
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    try:
        regressor = Regressor(regressor_class, *args, **kwargs)
        regressor.fit(train_x, train_y)
    except Exception as exc:
        print("Exception occured while building " + str(regressor_class))
        print(str(exc))
        raise
        return None, {"train": np.nan, "test": np.nan}, np.nan

    rmse = {}
    rmse["train"] = _rmse(regressor, train_x, train_y)
    print("Train RMSE:\n", rmse["train"])

    rmse["test"] = _rmse(regressor, test_x, test_y)
    print("Test RMSE:\n", rmse["test"])

    #scores = cross_val_score(
    #    regressor, train_x, train_y, scoring="neg_mean_squared_error", cv=10,
    #    n_jobs=-1)
    #rmse_scores = (-scores)**0.5
    rmse_scores = np.nan
    return regressor, rmse, rmse_scores


def _prepare_data(filter_func=None, keep_cols=None):
    """Prepare data for training and testing"""
    global df
    if filter_func is not None:
        filt_df = filter_func(df)
    else:
        filt_df = df
    # Split data into train and test
    train_set, test_set = train_test_split(
        filt_df, test_size=0.2, random_state=RANDOM_STATE)

    if keep_cols is None:
        drop_cols = [
            'Code du département', 'Libellé du département', 'Code de la commune',
            'Libellé de la commune', 'Abstentions', 'Blancs et nuls',
            'Choix A', 'Choix B', 'Choix A (%)', 'Choix B (%)', 'Abstentions (%)',
            'Blancs (%)']
        df_train = train_set.drop(columns=drop_cols)
        df_test = test_set.drop(columns=drop_cols)
    else:
        df_train = train_set[keep_cols].copy()
        df_test = test_set[keep_cols].copy()

    df_train_labels = train_set[
        ['Choix A (%)', 'Choix B (%)',
         'Abstentions (%)', 'Blancs (%)']].copy()
    df_test_labels = test_set[
            ['Choix A (%)', 'Choix B (%)',
             'Abstentions (%)', 'Blancs (%)']].copy()

    preproc_pline = Pipeline([
        # Fill missing values with the median
        ("imputer", SimpleImputer(strategy="median")),
        # Apply standard scaler to the data
        ("std_scaler", StandardScaler())])

#    preproc_pca_pline = Pipeline([
#        # Fill missing values with the median
#        ("imputer", SimpleImputer(strategy="median")),
#        # Apply standard scaler to the data
#        ("std_scaler", StandardScaler()),
#        ("pca", PCA(n_components=0.999))])

#    preproc_poly_pline = Pipeline([
#        # Fill missing values with the median
#        ("imputer", SimpleImputer(strategy="median")),
#        # Polynomial features
#        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
#        # Apply standard scaler to the data
#        ("std_scaler", StandardScaler())])

    df_train_tf = preproc_pline.fit_transform(df_train)
    df_test_tf = preproc_pline.transform(df_test)

    # df_train_poly_tf = preproc_poly_pline.fit_transform(df_train)
    # df_test_poly_tf = preproc_poly_pline.transform(df_test)

    data = {
        "default": {
            "train_x": df_train_tf,
            "train_y": df_train_labels,
            "test_x": df_test_tf,
            "test_y": df_test_labels
            },
#        "poly": {
#            "train_x": df_train_poly_tf,
#            "train_y": df_train_labels,
#            "test_x": df_test_poly_tf,
#            "test_y": df_test_labels
#            }
        }
#    for key_ in ["PolyLinearRegression", "PolySGDR_l1", "PolySGDR_l2"]:
#        data[key_] = data["poly"]
    x_keys = df_train.keys()
    y_keys = ['Choix A (%)', 'Choix B (%)', 'Abstentions (%)', 'Blancs (%)']
    return data, preproc_pline, x_keys, y_keys


def analyse_data(all_data=True):
    """Train a set of regressors and compute performance"""
    if all_data == True:
        filter_func = None
        keep_cols = None
    else:
        def filter_func(df):
            df = df.loc[df["Population en 2013 (princ)"] > 10e3]
            return df

        keep_cols = [
            'Longitude', 'Pop Hommes en 2013 (princ)',
            'Pop Femmes en 2013 (princ)',
            'Pop Femmes 60-74 ans en 2013 (princ)',
            'Pop Femmes 65 ans ou plus en 2013 (princ)',
            'Pop 1 an ou plus habitant 1 an avt autre région métropole en 2013 (princ)',
            'Pop 15-24 ans habitant 1 an avt autre logt en 2013 (princ)',
            'Pop 15 ans ou plus Femmes en 2013 (compl)',
            'Pop 15-24 ans Ouvriers en 2013 (compl)',
            'Pop 25-54 ans Autres en 2013 (compl)',
            'Pop 55 ans ou plus Employés en 2013 (compl)',
            'Pop 55 ans ou plus Ouvriers en 2013 (compl)',
            'Pop 55 ans ou plus Autres en 2013 (compl)',
            'Aucun diplôme ou au mieux BEPC, brevet des collèges, DNB\nHommes\n16 à 24 ans\nRP2010',
            'Diplôme de niveau CAP, BEP\nHommes\n25 ans ou plus\nRP2010',
            'Diplôme de niveau Baccalauréat (général, techno, pro)\nHommes\n16 à 24 ans\nRP2010',
            'Diplôme de niveau Baccalauréat (général, techno, pro)\nFemmes\n16 à 24 ans\nRP2010',
            "dont part des revenus d'activités non salariées",
            'Part des revenus du patrimoine et autres revenus',
            'Rapport inter-décile 9e décile/1er decile']
    data, preproc_pline, x_keys, y_keys = _prepare_data(
        filter_func=filter_func, keep_cols=keep_cols)

#    rndforest_grid = [
#        {"n_estimators": [10, 50, 100], "max_features": [1, 2, 3, 4, 5],
#         "max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [16]
#         }]
    regression_classes = [
        ("LinearRegression", LinearRegression, (), {}),
        ("SGDRegressor_l1", SGDRegressor, (), {"penalty": "l1"}),
        ("SGDRegressor_l2", SGDRegressor, (), {"penalty": "l2"}),
        ("ElasticNet", ElasticNet, (), {}),
        # ("PolyLinearRegression", LinearRegression, (), {}),
        # ("PolySGDR_l1", SGDRegressor, (), {"penalty": "l1"}),
        # ("PolySGDR_l2", SGDRegressor, (), {"penalty": "l2"})
        #
        # ("SVR", SVR, (), {}),
        # Random Forest without PCA
        ("RandomForest", RandomForestRegressor, (),
         {"n_estimators": 250, "n_jobs": -1, "verbose": 1,
          "min_samples_leaf": 16, "max_features": 16, "max_leaf_nodes": 256}), #{"n_estimators": 300, "max_leaf_nodes": 32, "n_jobs": -1}),
        # Random Forest with PCA
#         ("RandomForest", RandomForestRegressor, (),
#          {"n_estimators": 100, "min_samples_leaf": 16, "max_leaf_nodes": 64, "n_jobs": -1, "verbose": 1}),
        ("AdaBoost", AdaBoostRegressor, (), {}),
        ("GradientBoost", GradientBoostingRegressor, (),
         {"min_samples_leaf": 16, "max_depth": 10}),
#        ("MLPRegressor", MLPRegressor, (),
#         {"max_iter": 1000, "hidden_layer_sizes": (100,)*20, "early_stopping": True,
#          "activation": "relu",
#          "learning_rate": "constant", "verbose": True}),
#        ("GridSearchCV", GridSearchCV, (), {
#            "estimator": RandomForestRegressor(),
#            "param_grid": rndforest_grid,
#            "cv": 5, "scoring":"neg_mean_squared_error",
#            "n_jobs": -1})
        ]

    regressors = {}
    rmse = {}
    rmse_scores = {}
    # Iterate over all the regressors
    for (name_, cls_, fnargs_, fnkwargs_) in regression_classes:
        print(name_)
        data_ = data.get(name_, data["default"])
        regressors[name_], rmse[name_], rmse_scores[name_] = _build_regressor(
            data_, cls_, *fnargs_, **fnkwargs_)
    return regressors, rmse, rmse_scores, x_keys, y_keys, data


def load_regresults(filename=None):
    """Load regression results from pickle file"""
    if filename is None:
        filename = "results/regressors_rmse1.p"
    with io.open(filename, 'rb') as f:
        regresults = pickle.load(f)
    regressors = regresults["regressors"]
    rmse = regresults["rmse"]
    x_keys = regresults["x_keys"]
    y_keys = regresults["y_keys"]
    data = regresults["data"]
    return regressors, rmse, None, x_keys, y_keys, data


if __name__ == "__main__":
    # regressors, rmse, rmse_scores, x_keys, y_keys, data = analyse_data()
    regressors, rmse, rmse_scores, x_keys, y_keys, data = load_regresults()
    (big_regressors, big_rmse, big_rmse_scores,
     big_x_keys, big_y_keys, big_data) = (
        load_regresults("results/regressors_bigcities_impfeatures.p"))

    # with io.open("results/regressors_bigcities_impfeatures.p", 'wb') as f:
    #     pickle.dump(
    #         {"regressors": regressors, "rmse": rmse, "x_keys": x_keys,
    #          "y_keys": y_keys, "data": data}, f)
