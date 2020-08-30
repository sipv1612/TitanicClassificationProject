import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


class TitanicClassifier:
    def __init__(self,
                 numerical_data=None,
                 numerical_test_data=None,
                 numerical_labels=None,
                 categorical_data=None,
                 categorical_test_data=None,
                 categorical_labels=None):
        self.numerical_data = numerical_data
        self.numerical_test_data = numerical_test_data
        self.numerical_labels = numerical_labels
        self.categorical_data = categorical_data
        self.categorical_test_data = categorical_test_data
        self.categorical_labels = categorical_labels
        self.models = {}

    def __add_model(self, model, model_name, data_type='numerical', result_type='binary'):
        if model_name is not None:
            if model_name in self.models:
                self.models[model_name]['model'] = model
                self.models[model_name]['data_type'] = data_type
                self.models[model_name]['result_type'] = result_type
                return None
            else:
                self.models[model_name] = {'model': model, 'data_type': data_type, 'result_type': result_type}
                return None
        else:
            return model

    def add_sklearn_mlpc_model(self,
                               model_name='sklearn_mlpc',
                               layers_shape=(8, 4, 2, 1),
                               solver='lbfgs',
                               activation='relu',
                               alpha=0.0001,
                               random_state=1,
                               batch_size=32,
                               max_iter=10000):
        model = MLPClassifier(
            solver=solver,
            alpha=alpha,
            hidden_layer_sizes=layers_shape,
            random_state=random_state,
            activation=activation,
            batch_size=batch_size,
            max_iter=max_iter)
        model.fit(self.numerical_data, self.numerical_labels)
        return self.__add_model(model, model_name)

    def add_keras_mlpc_model(self,
                             model_name='keras_mlpc',
                             layers_shape=[8, 4, 2, 1],
                             layers_activation_funcs=['relu', 'sigmoid', 'sigmoid', 'sigmoid'],
                             loss_func='binary_crossentropy',
                             optimizer=RMSprop(0.001),
                             epochs=1000,
                             batch_size=32,
                             report_every=100,
                             metrics=['accuracy']):
        if len(layers_shape) != len(layers_activation_funcs):
            print('Input layers shape and activation funcs dimension not match!')
            return None
        model = Sequential()
        Dense(layers_shape[0], activation=layers_activation_funcs[0], input_shape=[self.numerical_data.shape[1]])
        for index in range(1, len(layers_shape)):
            model.add(Dense(layers_shape[index], activation=layers_activation_funcs[index]))
        model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)
        model.fit(
            x=self.numerical_data, y=self.numerical_labels,
            epochs=epochs, verbose=0, batch_size=batch_size, callbacks=[tfdocs.modeling.EpochDots(report_every)])

        return self.__add_model(model, model_name)

    def add_sklearn_random_forest(self,
                                  model_name='sklearn_rdf',
                                  n_estimators=100,
                                  criterion='gini',
                                  max_depth=4,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features='auto',
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None,
                                  bootstrap=True,
                                  oob_score=False,
                                  n_jobs=None,
                                  random_state=None,
                                  verbose=0,
                                  warm_start=False,
                                  class_weight=None,
                                  ccp_alpha=0.0,
                                  max_samples=None):
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       criterion=criterion,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf,
                                       max_features=max_features,
                                       max_leaf_nodes=max_leaf_nodes,
                                       min_impurity_decrease=min_impurity_decrease,
                                       min_impurity_split=min_impurity_split,
                                       bootstrap=bootstrap,
                                       oob_score=oob_score,
                                       n_jobs=n_jobs,
                                       random_state=random_state,
                                       verbose=verbose,
                                       warm_start=warm_start,
                                       class_weight=class_weight,
                                       ccp_alpha=ccp_alpha,
                                       max_samples=max_samples)
        model.fit(self.numerical_data, self.numerical_labels)
        self.__add_model(model, model_name)

    def predict(self, model_name, threshold=0.5, data=None):
        if data is None:
            if self.models[model_name]['data_type'] == 'numerical':
                predictions = self.models[model_name]['model'].predict(self.numerical_test_data)
                if self.models[model_name]['result_type'] == 'numerical':
                    predictions = np.where(predictions > threshold, 1, 0).T[0]
                return predictions
            elif self.models[model_name]['data_type'] == 'categorical':
                return self.models[model_name]['model'].predict(self.categorical_test_data)
                rounded = np.where(predictions > threshold, 1, 0).T[0]
            else:
                return None
        else:
            if self.models[model_name]['data_type'] == 'numerical':
                predictions = self.models[model_name]['model'].predict(data)
                rounded = np.where(predictions > threshold, 1, 0).T[0]
                return rounded
            elif self.models[model_name]['data_type'] == 'categorical':
                return self.models[model_name]['model'].predict(data)
            else:
                return None
