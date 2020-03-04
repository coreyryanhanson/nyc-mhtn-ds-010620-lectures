import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class ModelSwitcher(object):
    def __init__(self, df, target, selection):
        self.random_state = 1
        self.test_size = .2
        self.selection = selection
        self.target = target
        self.X = df[selection]
        self.y = df[target]
        self._train_test_split()
        self._check_balance()
        self._scale_setter()
        self._scale_getter()

    def _train_test_split(self):
        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def _scale_setter(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)

    def _scale_getter(self):
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _check_balance(self):
        new_X, new_y = self._class_imbalance(self.target, 0, 1)
        self.X_train, self.y_train = new_X, new_y

    def _class_imbalance(self, target, majority_val, minority_val):
        df = pd.concat([self.X_train, self.y_train], axis=1)
        majority, minority = df[df[target] == majority_val], df[df[target] == minority_val]
        simp_upsample = self._simple_resample(minority, majority)
        simp_downsample = self._simple_resample(majority, minority)
        smote_x, smote_y = self._smote_data(self.X_train, self.y_train)
        up_y, up_X = simp_upsample[target], simp_upsample.drop(target, axis=1)
        return up_X, up_y

    def _simple_resample(self, to_change, goal):
        resampled = resample(to_change, replace=True, n_samples=len(goal), random_state=self.random_state)
        return pd.concat([goal, resampled])

    def _smote_data(self, X_train, y_train):
        sm = SMOTE(sampling_strategy=1.0, random_state=self.random_state)
        sm.fit_sample(X_train, y_train)
        return X_train, y_train


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
