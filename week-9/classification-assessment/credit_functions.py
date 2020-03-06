import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import pickle
import inspect


class ModelSwitcher(object):
    def __init__(self, data):
        if type(data) == DataPreprocessor:
            self.data = data
        else:
            pass

    def instantiate_data(self, df, target, cat_features=[], cont_features=[], balance_class=False, scale_type=False, poly_degree=False):
        self.data = DataPreprocessor(df, target, cat_features=cat_features, cont_features=cont_features, balance_class=balance_class, scale_type=scale_type, poly_degree=poly_degree)

class DataPreprocessor(object):
    def __init__(self, df, target, cat_features=[], cont_features=[], balance_class=False, scale_type=False, poly_degree=False):
        self.df = df
        self.random_state = 1
        self.test_size = .2
        self.poly_degree = poly_degree
        self.balance_class = balance_class
        self.scale_type = scale_type
        self.cat_features = pd.Index(cat_features)
        self.cont_features = pd.Index(cont_features)
        self.init_selection = self.cont_features.union(self.cat_features, sort=False)
        self.selection = self.init_selection
        self.target = target
        self.X = df[self.selection]
        self.y = df[target]
        self._data_preprocessing()


    def _train_test_split(self):
        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def _scale_setter(self):
        if self.scale_type == "standard":
            print("Using standard scaler")
            self.scaler = StandardScaler()
        elif self.scale_type == "minmax":
            print("Using Min/Max scaler")
            self.scaler = MinMaxScaler()
        else:
            print("No scaling specified")
            self.scale_type = False
            return
        self.scaler.fit(self.X_train)

    def _scale_getter(self):
        if self.scale_type == False:
            print("Skipping scaling")
            return
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _data_preprocessing(self):
        self._poly_features()
        self._train_test_split()
        self._class_imbalance(0, 1)
        self._scale_setter()
        self._scale_getter()

    def _class_imbalance(self, majority_val, minority_val):
        target = self.target
        df = pd.concat([self.X_train, self.y_train], axis=1)
        majority, minority = df[df[target] == majority_val], df[df[target] == minority_val]
        if self.balance_class == "upsample":
            print("Performing upsample")
            self._simple_resample(minority, majority)
        elif self.balance_class == "downsample":
            print("Performing downsample")
            self._simple_resample(majority, minority)
        elif self.balance_class == "smote":
            print("Performing SMOTE")
            self._smote_data()
        elif self.balance_class == "tomek":
            print("Performing Tomek Links")
            self._tomek_data()
        else:
            print("Skipping class imbalance functions")

    def _simple_resample(self, to_change, goal):
        resampled = resample(to_change, replace=True, n_samples=len(goal), random_state=self.random_state)
        joined = pd.concat([goal, resampled])
        self.X_train, self.y_train = joined.drop(self.target, axis=1), joined[self.target]

    def _smote_data(self):
        sm = SMOTE(sampling_strategy=1.0, random_state=self.random_state)
        self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

    def _tomek_data(self):
        tl = TomekLinks()
        self.X_train, self.y_train = tl.fit_sample(self.X_train, self.y_train)

    def _poly_features(self):
        if type(self.poly_degree) == int:
            print(f"Getting polynomial features of degree {self.poly_degree}")
            X_cont = self.X[self.cont_features]
            X_cont_index = X_cont.index
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_cont)
            columns = pd.Index(poly.get_feature_names(X_cont.columns))
            poly_df = pd.DataFrame(X_poly, index=X_cont_index, columns=columns)
            self.poly_columns = columns.drop(labels=self.cont_features)
            self.X = pd.concat([self.X[self.init_selection], poly_df[self.poly_columns]], axis=1)
            self.selection = self.init_selection.union(self.poly_columns, sort=False)
        else:
            print("Skipping polynomial features")
            self.poly_degree = False
            self.X = self.X[self.init_selection]

    def column_drop(self, columns):
        self.selection = self.selection.drop(labels=columns)


def pickle_read(path):
    with open(path, "rb") as f:
        pickle_file = pickle.load(f)
    return pickle_file

def pickle_write(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)

def increasing_debt(row, column, i):
    if i > 1 and row[column + f'{i}'] < row[column + f'{i - 1}'] and row["is_streak"] == 1:
        row["debt_streak"] += 1
        row["raw_debt_accum"] += row[column + f'{i - 1}'] - row[column + f'{i}']
    else:
        row["is_streak"] = 0
    return row


def initiate_placeholders(df):
    df["is_streak"], df["debt_streak"] = 1, 0
    df["raw_debt_accum"] = 0
    return df


def remove_placeholders(df):
    return df.drop(columns=["is_streak", "raw_debt_accum"])


def replace_unknowns(df):
    education_dict = {4: 0, 5: 0, 6: 0}
    marriage_dict = {3: 0}
    df["EDUCATION"].replace(education_dict, inplace=True)
    df["MARRIAGE"].replace(marriage_dict, inplace=True)
    return df


# Gathers column names to exclude
def exclude_columns(looped_cols):
    looped_exc = []
    for col in looped_cols:
        sing_exc = [col + f"{i}" for i in np.arange(1, 7)]
        looped_exc.extend(sing_exc)
    looped_exc.extend(["ID", "default payment next month"])
    return looped_exc


def calculate_utilization(df):
    df["avg_utilization"], df["avg_payment_impact"] = 0, 0
    initiate_placeholders(df)
    for i in np.arange(1, 7):
        df['payment_impact' + f'{i}'] = (df['PAY_AMT' + f'{i}']) / df["LIMIT_BAL"]
        df["utilization" + f'{i}'] = df["BILL_AMT" + f'{i}'] / df["LIMIT_BAL"]
        if i > 1:
            df = df.apply(lambda x: increasing_debt(x, "utilization", i), axis=1)
        df["avg_utilization"] += df["utilization" + f'{i}']
        df["avg_payment_impact"] += df["payment_impact" + f'{i}']
    df["avg_utilization"] = df["avg_utilization"] / 6
    df["avg_payment_impact"] = df["avg_payment_impact"] / 6
    df["debt_avg_delta"] = (df["raw_debt_accum"] / df["debt_streak"]).fillna(0)
    df = remove_placeholders(df)
    return df

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
