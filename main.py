import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_transform import DataTransform
from titanic_classifier import TitanicClassifier


if __name__ == '__main__':
    train_data = DataTransform(pd.read_csv('train.csv'), 'Survived')
    test_data = DataTransform(pd.read_csv('test.csv'))
    train_data.numerical_transform(inplace=True, drop_columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])
    test_data.numerical_transform(inplace=True, drop_columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])
    scaler = StandardScaler()
    scaler.fit(train_data.get_features() + test_data.get_features())
    train_data.apply_scaler(scaler)
    test_data.apply_scaler(scaler)

    classifier = TitanicClassifier(
        numerical_data=train_data.get_features(),
        numerical_test_data=test_data.get_features(),
        numerical_labels=train_data.get_labels())

    #~78%
    classifier.add_sklearn_random_forest(model_name='sklearn_rdf')

    #~62%
    # classifier.add_keras_mlpc_model(model_name='keras_mlpc', epochs=10000, report_every=250)

    #not working yet
    # classifier.add_sklearn_mlpc_model(model_name='sklearn_mlpc')

    predictions = classifier.predict(model_name='sklearn_rdf', threshold=0.5)
    predictions = pd.DataFrame({'PassengerId': test_data.get_raw_features()['PassengerId'], 'Survived': predictions})
    predictions.Survived = predictions.Survived.astype(int)
    predictions.to_csv('predicts.csv', index=False)
