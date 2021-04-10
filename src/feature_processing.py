from src import config
import  pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
def data_processin_H1(data, mode='train'):
    df = pd.read_csv(data)
    X = df[df.columns.difference(['quality'])]
    y = df['quality']
    y.to_csv(config.LABEL_DATA,index = False)

    def castom_standardscaler_label_encoder(X, numeric_features=None, categoric_features=None, mode ="train"):
        """
        This function takes data, numeric_features[list] and categoric_features[list] and returns changed data.

        For numeric_features: Data -> Fillna(SimpleImputer(strategy='mean')) -> Scale(StandardScaler) -> Data

        For categoric_features: Data -> OneHotEncoder -> Data

        """
        if categoric_features is None:
            categoric_features = []
        if numeric_features is None:
            numeric_features = []
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        import numpy as np
        import warnings
        import pandas as pd
        from pandas.core.common import SettingWithCopyWarning

        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        labelencoders = {}
        if numeric_features and categoric_features:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='median').fit(X.loc[ : , numeric_features])
            X[numeric_features] = imp_mean.transform(X.loc[ : , numeric_features])

            scale = StandardScaler().fit(X.loc[ : , numeric_features])
            newX = scale.transform(X.loc[ : , numeric_features])
            X_scaled = pd.DataFrame(newX, columns=numeric_features)
            if mode == 'train':
                for c in categoric_features:
                    labelencoders[c] = LabelEncoder()
                    X[c] = labelencoders[c].fit_transform(X[c])
            else:
                for c in categoric_features:
                    labelencoders = joblib.load(config.MODELS_PATH+"feature_encoders.pkl")
                    X[c] = labelencoders[c].transform(X[c])
            X_hot = X[categoric_features]

            return (pd.concat([X_hot, X_scaled], axis=1),labelencoders)

        elif numeric_features and not categoric_features:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='median').fit(X[numeric_features])
            X[numeric_features] = imp_mean.transform(X[numeric_features])

            scale = StandardScaler().fit(X[numeric_features])
            newX = scale.transform(X[numeric_features])
            X_scaled = pd.DataFrame(newX, columns=numeric_features)

            return (pd.concat([X[X.columns.difference(numeric_features)], X_scaled], axis=1), labelencoders)

        elif categoric_features and not numeric_features:
            if mode == 'train':
                for c in categoric_features:
                    labelencoders[c] = LabelEncoder()
                    X[c] = labelencoders[c].fit_transform(X[c])
            else:
                for c in categoric_features:
                    labelencoders = joblib.load(config.MODELS_PATH+"feature_encoders.pkl")
                    X[c] = labelencoders[c].transform(X[c])
            X_hot = X[categoric_features]

            return (pd.concat([X[X.columns.difference(categoric_features)], X_hot], axis=1), labelencoders)

        return (X, labelencoders)

    categoric_features = ['type']
    numeric_features = ['alcohol', 'chlorides', 'citric acid', 'density', 'fixed acidity',
           'free sulfur dioxide', 'pH', 'residual sugar', 'sulphates',
           'total sulfur dioxide', 'volatile acidity']
    X, labelencoders = castom_standardscaler_label_encoder(X, numeric_features = numeric_features, categoric_features = categoric_features,mode='train')

    X.to_csv(config.PROCESSED_X_H1,index = False)

    joblib.dump(labelencoders,config.MODELS_PATH+'feature_encoders_H1.pkl')

def data_processin_H2(data, mode='train'):
    df = pd.read_csv(data)
    X = df[df.columns.difference(['quality'])]
    y = df['quality']
    y.to_csv(config.LABEL_DATA,index = False)

    pd.options.mode.chained_assignment = None

    X.drop(X[['type']], axis=1, inplace=True)
    X['free sulfur dioxide*total sulfur dioxide'] = X['free sulfur dioxide'] * X['total sulfur dioxide']
    X.drop(X[['free sulfur dioxide', 'total sulfur dioxide']], axis=1, inplace=True)
    numeric_features = ['alcohol', 'chlorides', 'citric acid', 'density', 'fixed acidity', 'pH',
                        'residual sugar', 'sulphates', 'volatile acidity',
                        'free sulfur dioxide*total sulfur dioxide']
    def castom_standardscaler_label_encoder(X, numeric_features=None, categoric_features=None, mode ="train"):
        """
        This function takes data, numeric_features[list] and categoric_features[list] and returns changed data.

        For numeric_features: Data -> Fillna(SimpleImputer(strategy='mean')) -> Scale(StandardScaler) -> Data

        For categoric_features: Data -> OneHotEncoder -> Data

        """
        if categoric_features is None:
            categoric_features = []
        if numeric_features is None:
            numeric_features = []
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        import numpy as np
        import warnings
        import pandas as pd
        from pandas.core.common import SettingWithCopyWarning

        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        labelencoders = {}
        if numeric_features and categoric_features:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='median').fit(X.loc[ : , numeric_features])
            X[numeric_features] = imp_mean.transform(X.loc[ : , numeric_features])

            scale = StandardScaler().fit(X.loc[ : , numeric_features])
            newX = scale.transform(X.loc[ : , numeric_features])
            X_scaled = pd.DataFrame(newX, columns=numeric_features)
            if mode == 'train':
                for c in categoric_features:
                    labelencoders[c] = LabelEncoder()
                    X[c] = labelencoders[c].fit_transform(X[c])
            else:
                for c in categoric_features:
                    labelencoders = joblib.load(config.MODELS_PATH+"feature_encoders.pkl")
                    X[c] = labelencoders[c].transform(X[c])
            X_hot = X[categoric_features]

            return (pd.concat([X_hot, X_scaled], axis=1),labelencoders)

        elif numeric_features and not categoric_features:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='median').fit(X[numeric_features])
            X[numeric_features] = imp_mean.transform(X[numeric_features])

            scale = StandardScaler().fit(X[numeric_features])
            newX = scale.transform(X[numeric_features])
            X_scaled = pd.DataFrame(newX, columns=numeric_features)

            return (pd.concat([X[X.columns.difference(numeric_features)], X_scaled], axis=1), labelencoders)

        elif categoric_features and not numeric_features:
            if mode == 'train':
                for c in categoric_features:
                    labelencoders[c] = LabelEncoder()
                    X[c] = labelencoders[c].fit_transform(X[c])
            else:
                for c in categoric_features:
                    labelencoders = joblib.load(config.MODELS_PATH+"feature_encoders.pkl")
                    X[c] = labelencoders[c].transform(X[c])
            X_hot = X[categoric_features]

            return (pd.concat([X[X.columns.difference(categoric_features)], X_hot], axis=1), labelencoders)

        return (X, labelencoders)

    X, labelencoders = castom_standardscaler_label_encoder(X, numeric_features = numeric_features,mode='train')

    X.to_csv(config.PROCESSED_X_H2,index = False)

    joblib.dump(labelencoders,config.MODELS_PATH+'feature_encoders_H2.pkl')

def data_processin_H3(data, mode='train'):
    df = pd.read_csv(data)

    pd.options.mode.chained_assignment = None

    def remove_duplicates(data):
        """
        This function removes duplicates from dataset
        :return : returns new data
        """

        print('There are', str(data.duplicated().sum()), 'duplicated records.')
        new = data.drop_duplicates()
        print('There were', f'{(1 - new.shape[0] / data.shape[0]) * 100:.2f}',
              '% duplicates in dataset. It was removed.')

        return new

    def fill_missing(data, method='median'):
        '''
        This function fill NaNs with means by each type and quality
        :return: data
        '''
        main_features = data.columns.difference(['type', 'quality'])
        for color, df_by_type in data.groupby('type'):
            for category, df_by_quality in df_by_type.groupby('quality'):
                median = df_by_quality[main_features].median()
                mean = df_by_quality[main_features].mean()
                data[(data['type'] == color) & (data['quality'] == category)] = data[
                    (data['type'] == color) & (data['quality'] == category)].fillna(
                    median if method == 'median' else mean)

        return data

    df = remove_duplicates(df)
    df = fill_missing(df)
    categoric_features = ['type']
    labelencoders = {}
    if mode == 'train':
        for c in categoric_features:
            labelencoders[c] = LabelEncoder()
            df[c] = labelencoders[c].fit_transform(df[c])
    else:
        for c in categoric_features:
            labelencoders = joblib.load(config.MODELS_PATH + "feature_encoders_H3.pkl")
            df[c] = labelencoders[c].transform(df[c])

    X = df[df.columns.difference(['quality'])]
    y = df['quality']
    y.to_csv(config.LABEL_DATA, index=False)
    X.to_csv(config.PROCESSED_X_H3,index = False)

    joblib.dump(labelencoders,config.MODELS_PATH+'feature_encoders_H3.pkl')

data_processin_H1(config.DATA)
# data_processin_H2(config.DATA)
# data_processin_H3(config.DATA)
