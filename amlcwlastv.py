!pip install keras-tuner
!pip install tensorflow-gpu


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy

from numpy import set_printoptions
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import kerastuner as kt
from sklearn import metrics
from sklearn import model_selection

#imports dataset
filename = 'train_imperson_without4n7_balanced_data.csv'
filename2 = 'test_imperson_without4n7_balanced_data.csv'
dataframe = read_csv(filename)
dataframe_test = read_csv(filename2)
dataframe.dropna()
dataframe_test.dropna()
array = dataframe.values
arraytest = dataframe_test.values
X_train = array[:,0:152]
Y_train = array[:,152]

X_test = arraytest[:,0:152]
Y_test = arraytest[:,152]

#create SAE additional features
k_reg = tf.keras.regularizers.L1(0)
b_reg = tf.keras.regularizers.L1(0)
a_reg = tf.keras.regularizers.L1(0)

encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg),
    # tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg)
])
decoder = tf.keras.Sequential([
    # tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg),
    tf.keras.layers.Dense(152, activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg)
])
auto_encoder = tf.keras.Sequential([encoder, decoder])
auto_encoder.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)
auto_encoder.fit(X_train, X_train, epochs=200, validation_split=0.2, verbose=1, callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3))

#x train with sae features (+16)
x_hat = encoder.predict(X_train)
X_with_auto = np.concatenate([X_train, x_hat], 1)

#x test with sae features (+16)
x_test_hat = encoder.predict(X_test)
X_test_with_auto = np.concatenate([X_test, x_test_hat], 1)

#add PCA features
pca_9 = PCA(n_components= 9)
pca_9.fit(X_train)

#x train with sae features (+16) and pca features (+9)
x_pca = pca_9.transform(X_train)
x_with_new_features = np.concatenate([X_with_auto, x_pca], 1)

#x test with sae features (+16) and pca features (+9)
x_pca_test = pca_9.transform(X_test)
x_with_new_features_test=np.concatenate([X_test_with_auto, x_pca_test], 1)

#define a function to fintune hyperparameters
def build_model(hp):
    inputs = tf.keras.Input(shape=(10))
    x = inputs
    k_reg = tf.keras.regularizers.L1(0.001)
    b_reg = tf.keras.regularizers.L1(0.001)
    a_reg = tf.keras.regularizers.L1(0.001)
    for i in range(hp.Int('dense_blocks', 3, 5, default=3)):
        x = tf.keras.layers.Dense(
            hp.Int('hidden_size_' + str(i), 30, 100, step=10, default=50),
            activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg)(x)
        x = tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    x = tf.keras.layers.Dense(
        hp.Int('hidden_size_' + str(i + 1), 30, 100, step=10, default=50),
        activation='relu', kernel_regularizer=k_reg, bias_regularizer=b_reg, activity_regularizer=a_reg)(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def tune_hyper_parameters(x_train, y_train, project_name):
    # tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=3, hyperband_iterations=2, project_name=project_name)
    tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=30, hyperband_iterations=2, project_name=project_name)
    # tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=2)
    tuner.search(x_train,
                 y_train,
                 validation_split=0.2,
                #  epochs=1,
                 epochs=20,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
    return tuner

def build_linear_model(hp):
    model = LogisticRegression(penalty='elasticnet',
                               C=hp.Float('reg', 1e-3, 1, 1e-4),
                               solver='saga',
                               l1_ratio=hp.Float('l1_ratio', 1e-3, 0.9999, 1e-4))
    return model

def tune_linear_hyper_parameters(x_train, y_train, project_name):
    tuner = kt.tuners.Sklearn(
            oracle=kt.oracles.BayesianOptimization(
                objective=kt.Objective('score', 'max'),
                max_trials=30),
            hypermodel=build_linear_model,
            scoring=metrics.make_scorer(metrics.accuracy_score),
            cv=model_selection.StratifiedKFold(5),
            directory='.',
            project_name=project_name)
    # tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=2)
    tuner.search(x_train, y_train)
    return tuner


#def function to find best features
def find_best_features(scores, n=10):
    return np.argsort(-1*scores)[:n]

def transform_features(x, scores):
    indexes = find_best_features(scores)
    return x[:, indexes]




#selected tree features
# feature extraction random forest
#this is the 3rd try
#here are the 3rd list of numbers
model = ExtraTreesClassifier()
model.fit(x_with_new_features, Y_train)
print(find_best_features(model.feature_importances_))

selected_x_train_from_tree = transform_features(x_with_new_features, model.feature_importances_)
selected_x_test_tree = transform_features(x_with_new_features_test, model.feature_importances_)
tuner = tune_linear_hyper_parameters(selected_x_train_from_tree, Y_train, "linear_selected_tree_features")
best_selected_by_tree_params = tuner.get_best_hyperparameters(1)[0]
print(best_selected_by_tree_params.values)
best_model = tuner.get_best_models(1)[0].fit(selected_x_train_from_tree, Y_train)
print("best logistic regression model from selected tree feature", best_model.score(selected_x_test_tree, Y_test))
confusion_matrix(best_model.predict(selected_x_test_tree), Y_test)
tuner = tune_hyper_parameters(selected_x_train_from_tree, Y_train, "selected_tree_features")
best_selected_by_tree_params = tuner.get_best_hyperparameters(1)[0]
print(best_selected_by_tree_params.values)
best_model = tuner.get_best_models(1)[0]
best_model.fit(selected_x_train_from_tree, Y_train)
print("best model from selected tree feature", best_model.evaluate(selected_x_test_tree, Y_test))
print(confusion_matrix(np.round(best_model.predict(selected_x_test_tree)), Y_test))


#selected K-best features
test = SelectKBest(score_func=mutual_info_classif, k=10)
sk_fit = test.fit(x_with_new_features, Y_train)
print(sk_fit.get_support(True))

sk_features_train = sk_fit.transform(x_with_new_features)
sk_features_test = sk_fit.transform(x_with_new_features_test)
tuner = tune_linear_hyper_parameters(sk_features_train, Y_train, "linear_selected_k_features")
best_selected_by_tree_params = tuner.get_best_hyperparameters(1)[0]
print(best_selected_by_tree_params.values)
best_model = tuner.get_best_models(1)[0]
best_model.fit(sk_features_train, Y_train)
print("best linear model from selected k best feature", best_model.score(sk_features_test, Y_test))
print(confusion_matrix(best_model.predict(sk_features_test), Y_test))
tuner = tune_hyper_parameters(sk_features_train, Y_train, "selected_k_features")
best_selected_by_tree_params = tuner.get_best_hyperparameters(1)[0]
print(best_selected_by_tree_params)
best_model = tuner.get_best_models(1)[0]
best_model.fit(sk_features_train, Y_train)
print("best model from selected k best feature", best_model.evaluate(sk_features_test, Y_test))
print(confusion_matrix(np.round(best_model.predict(sk_features_test)), Y_test))



#selected FRE features
#for recursive feature elimination
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 10)
rfe.fit(x_with_new_features, Y_train)
print(rfe.get_support(True))

selected_rfe_features_train = rfe.transform(x_with_new_features)
selected_rfe_features_test = rfe.transform(x_with_new_features_test)
tuner = tune_linear_hyper_parameters(selected_rfe_features_train, Y_train, "linear)selected_rfe_features")
best_selected_by_tree_params = tuner.get_best_hyperparameters(1)[0]
print(best_selected_by_tree_params.values)
best_model = tuner.get_best_models(1)[0]
best_model.fit(selected_rfe_features_train, Y_train)
print("best linear model from selected rfe feature", best_model.score(selected_rfe_features_test, Y_test))
print(confusion_matrix(best_model.predict(selected_rfe_features_test), Y_test))
tuner = tune_hyper_parameters(selected_rfe_features_train, Y_train, "selected_rfe_features")
best_selected_by_tree_params = tuner.get_best_hyperparameters(1)[0]
print(best_selected_by_tree_params.values)
best_model = tuner.get_best_models(1)[0]
print("best model from selected rfe feature", best_model.evaluate(selected_rfe_features_test, Y_test))
print(confusion_matrix(np.round(best_model.predict(selected_rfe_features_test)), Y_test))
