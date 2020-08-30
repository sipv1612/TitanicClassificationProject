import pandas as pd


class DataTransform:
    def __init__(self, train_data, label_column=None):
        if label_column is None:
            self.__raw_features = train_data
            self.__labels = None
        else:
            self.__raw_features = train_data.drop(label_column, axis=1)
            self.__labels = train_data[label_column].to_numpy()
        self.__features = self.__raw_features

    def get_raw_features(self):
        return self.__raw_features.copy()

    def get_features(self):
        return self.__features.copy()

    def get_labels(self):
        return self.__labels

    def numerical_transform(self, inplace=False, drop_columns=None):
        features = self.__raw_features.copy()
        features.Sex = (features.Sex == 'male').astype(float)
        features.Age.fillna(value=features.Age.mean(), inplace=True)
        features.Embarked.fillna('S', inplace=True)
        features.Embarked = pd.Categorical(features.Embarked)
        features.Embarked = features.Embarked.cat.codes
        features.Pclass = features.Pclass.astype(float)
        features.SibSp = features.SibSp.astype(float)
        features.Parch = features.Parch.astype(float)
        features.Embarked = features.Embarked.astype(float)
        features.Fare.fillna(features.Fare.mean(), inplace=True)
        if drop_columns is not None:
            features.drop(labels=drop_columns, axis=1, inplace=True)
        if inplace:
            self.__features = features
        else:
            return features, self.__labels

    def apply_scaler(self, sklearn_scaler):
        if sklearn_scaler is not None:
            self.__features = sklearn_scaler.transform(self.__features)
