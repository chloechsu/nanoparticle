#!/usr/bin/env python
# coding: utf-8


import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import itertools


spectra_train_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/y_new_train.joblib')
# spectra produced from random forests generative model
spectra_test_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/y_test.joblib')
# spectra created from solving maxwell's equations
labels_train_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/x_new_train.joblib').reset_index()
# data fed into random forests generative model to create spectra_train_set
labels_test_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/x_test.joblib').reset_index()
# labels used to solve maxwell's equations and produce spectra_test_set


# Training/Test set made completely from simulated data. I used this to make sure the random forest model wasn't doing
# artificially well because its training set was made up of data created by a random forests model
labels_train_maxwell_equations = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/x_train.joblib').reset_index()
spectra_train_maxwell_equations = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/y_train.joblib')



def drop_indicies(df, column, condition_to_drop, update_existing_file = True):
    df_condition = df[column] == condition_to_drop 
    """
    This function takes a pandas df as input and drops a series of rows depending on a specified condition. For example, use this
    function to search through x_train and drop all rows where the material is not gold. 
    
    df - pandas dataframe 
    column - STR the column of the dataframe you want to use to determine if a row should be dropped 
    condition_to_drop - choose condition_to_drop such that the expression evalutes to true for the condition you want dropped 
    (ie if I want to only have Au samples, my column would be Material_Au and my condition would be 0, so that when the 
    condition would be true if the material was not gold)
    update_existing_file - BOOL, determines if the df that is inputted to this function is updated or if a new df with only the
    columns that aren't dropped by this function 
    """
    indicies_to_drop_list = []
    for row in df_condition.index:
        if df_condition.iloc[row] == True:
            indicies_to_drop_list.append(row)
    
    if update_existing_file == True:
        df.drop(indicies_to_drop_list, inplace=update_existing_file)
        return indicies_to_drop_list
    if update_existing_file == False:
        df_new = df.drop(indicies_to_drop_list, inplace=update_existing_file)
        return (df_new, indicies_to_drop_list)



def convert_from_one_hot(df_as_array, dictionary):
    """
    This function takes a dataframe (represented as a numpy array) that has been set up as a one hot encoding of a
    categorical variable and converts it to a numerical categorical system (for example, the shape representation
    [0,0,1,0] would become [2]). It takes as input the mxn dimensional array and a dictionary with n entries and returns
    a mx1 dimensional array.

    :param df_as_array: The data you want to convert from one hot encoding. Must be a numpy array
    :param dictionary: A dictionary containing representations for each one hot encoding
    :return: mx1 array containing variables represented as numerical encodings
    """
    catagories_list = []
    for row in df_as_array:
        row_tuple = tuple(row)
        catagories_list.append(dictionary[row_tuple])
    catagories_list_array = np.asarray(catagories_list)
    return catagories_list_array


def Train_Random_Forests_Shape_Classification(model_type, training_spectra, training_labels, 
                                              test_spectra, test_labels, convert_from_one_hot_dict, trees = 100,
                                              depth = None, m_features = 'auto'):
    """
    Trains random forest classifier to predict shape from nanoparticle emissivity spectra. Has the ability to train a
    model for all materials or one individial material

    :param model_type: specify whether this model will be trained on all materials or one individually. STRING. Options
    are "All", "SiO2", "SiN" and "Au"
    :param training_spectra - spectra to be used in training. Numpy Array
    :param training_spectra - labels to be used in training. Pandas Dataframe
    :param training_spectra - spectra to be used for testing. Numpy Array
    :param training_spectra - labels to be used for testing. Pandas Dataframe
    :param trees - number of trees used when creating the model. Default is 100
    :param convert_from_one_hot_dict - dictionary to convert one hot encoded representations to one dimensional
    representations
    :return: list containing: 1) accuracy of the model 2) the model itself 3) the confusion matrix for the model
    4) the output when the model is tested on the test set 5) the labels used in testing 6) the spectra used in testing
    """

    if model_type == 'All':
        print("training shape classifier all materials")
        labels_train_shape = training_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim', 
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2', 'index'] )
        labels_test_shape = test_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim', 
                                                        'Material_Au','Material_SiN', 'Material_SiO2', 'index'] )

        labels_train_shape_as_array = labels_train_shape.to_numpy()
        labels_test_shape_as_array = labels_test_shape.to_numpy()
        
        labels_train_shape_as_array_wo_OHE = convert_from_one_hot(labels_train_shape_as_array, convert_from_one_hot_dict)
        labels_test_shape_as_array_wo_OHE = convert_from_one_hot(labels_test_shape_as_array, convert_from_one_hot_dict)
        spectra_test_shape_as_array = test_spectra
        rf_model = RandomForestClassifier(n_estimators = trees, max_depth=depth, max_features=m_features)
        rf_model.fit(training_spectra, labels_train_shape_as_array_wo_OHE)
        accuracy = rf_model.score(test_spectra, labels_test_shape_as_array_wo_OHE)
        predictions = rf_model.predict(test_spectra)
        cm_rf = confusion_matrix(labels_test_shape_as_array_wo_OHE, predictions)
    
    if model_type == 'Au':
        labels_train_shape = training_labels.drop(columns = ['index','Material_SiO2','log Area/Vol', 'ShortestDim',
                                                          'MiddleDim', 'LongDim', 'Material_SiN'] )
        labels_test_shape=test_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim',
                                                    'Material_SiO2', 'Material_SiN'] )
        indicies_to_drop_train_list = drop_indicies(labels_train_shape, 'Material_Au', 0, True)
        indicies_to_drop_test_list = drop_indicies(labels_test_shape, 'Material_Au', 0, True)

        labels_train_shape.drop(columns = ['Material_Au'] , inplace=True)
        labels_test_shape.drop(columns = ['Material_Au'] , inplace=True)

        
    if model_type == 'SiN':
        labels_train_shape = training_labels.drop(columns = ['index','Material_SiO2','log Area/Vol', 'ShortestDim',
                                                          'MiddleDim', 'LongDim', 'Material_Au'] )
        labels_test_shape=test_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim',
                                                    'Material_SiO2', 'Material_Au'] )
        indicies_to_drop_train_list = drop_indicies(labels_train_shape, 'Material_SiN', 0, True)
        indicies_to_drop_test_list = drop_indicies(labels_test_shape, 'Material_SiN', 0, True)

        labels_train_shape.drop(columns = ['Material_SiN'] , inplace=True)
        labels_test_shape.drop(columns = ['Material_SiN'] , inplace=True)

    if model_type == 'SiO2':
        labels_train_shape = training_labels.drop(columns = ['index','Material_SiN','log Area/Vol', 'ShortestDim',
                                                          'MiddleDim', 'LongDim', 'Material_Au'] )
        labels_test_shape=test_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim',
                                                    'Material_SiN', 'Material_Au'] )
        indicies_to_drop_train_list = drop_indicies(labels_train_shape, 'Material_SiO2', 0, True)
        indicies_to_drop_test_list = drop_indicies(labels_test_shape, 'Material_SiO2', 0, True)

        labels_train_shape.drop(columns = ['Material_SiO2'] , inplace=True)
        labels_test_shape.drop(columns = ['Material_SiO2'] , inplace=True)
        
    if model_type in ['Au','SiN','SiO2']:
        print("training " + model_type)
        spectra_train_df = pd.DataFrame(training_spectra)
        spectra_test_df = pd.DataFrame(test_spectra)

        spectra_train_df.drop(indicies_to_drop_train_list, inplace=True)
        spectra_test_df.drop(indicies_to_drop_test_list, inplace=True)

        labels_train_shape_as_array = labels_train_shape.to_numpy()
        labels_test_shape_as_array = labels_test_shape.to_numpy()

        labels_train_shape_as_array_wo_OHE = convert_from_one_hot(labels_train_shape_as_array, convert_from_one_hot_dict)
        labels_test_shape_as_array_wo_OHE =  convert_from_one_hot(labels_test_shape_as_array, convert_from_one_hot_dict)

        spectra_train_shape_as_array = spectra_train_df.to_numpy()
        spectra_test_shape_as_array= spectra_test_df.to_numpy()

        rf_model = RandomForestClassifier(n_estimators = trees, max_depth=depth, max_features=m_features)
        rf_model.fit(spectra_train_shape_as_array, labels_train_shape_as_array_wo_OHE)
        accuracy = rf_model.score(spectra_test_shape_as_array, labels_test_shape_as_array_wo_OHE)

        predictions = rf_model.predict(spectra_test_shape_as_array)
        cm_rf = confusion_matrix(labels_test_shape_as_array_wo_OHE, predictions)

    return (accuracy, rf_model, cm_rf, predictions, labels_test_shape_as_array_wo_OHE, spectra_test_shape_as_array)



def Train_Random_Forests_Size_Regression(model_type, training_spectra, training_labels, 
                                              test_spectra, test_labels, trees = 10, m_features = 'auto'):

    """
    Trains random forest regressor to predict size from nanoparticle emissivity spectra. Has the ability to train a
    model for dimensions, volume, and both

    :param model_type: specify whether this model will be trained on size, dimensions, or both. STRING. Options
    are "volume", "all" and "size"
    :param training_spectra - spectra to be used in training. Numpy Array
    :param training_spectra - labels to be used in training. Pandas Dataframe
    :param training_spectra - spectra to be used for testing. Numpy Array
    :param training_spectra - labels to be used for testing. Pandas Dataframe
    :param trees - number of trees used when creating the model. Default is 10
    :return: list containing: 1) accuracy of the model 2) the model itself
    """



    if model_type == 'volume':
        labels_train = training_labels.drop(columns = ['ShortestDim', 'MiddleDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                            'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        labels_test = test_labels.drop(columns = ['ShortestDim', 'MiddleDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        rf_model = RandomForestRegressor(n_estimators = trees, n_jobs = -1, max_features=m_features)
        rf_model.fit(training_spectra, np.ravel(labels_train))
        accuracy = rf_model.score(test_spectra, np.ravel(labels_test))
    
    if model_type == 'all':
        labels_train = training_labels.drop(columns = ['Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                            'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        labels_test = test_labels.drop(columns = ['Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                            'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        labels_train_as_array = np.asarray(labels_train)
        labels_test_as_array = np.asarray(labels_test)

        rf_model_temp = RandomForestRegressor(n_estimators = trees)
        rf_model = MultiOutputRegressor(rf_model_temp, n_jobs = -1)
        rf_model.fit(training_spectra, labels_train_as_array)
        accuracy = rf_model.score(test_spectra, labels_test_as_array)
    
    if model_type == 'size':
        labels_train = training_labels.drop(columns = ['Geometry_TriangPrismIsosc', 'log Area/Vol',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                            'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        labels_test = test_labels.drop(columns = ['Geometry_TriangPrismIsosc', 'log Area/Vol',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        labels_train_as_array = np.asarray(labels_train)
        labels_test_as_array = np.asarray(labels_test)

        rf_model_temp = RandomForestRegressor(n_estimators = trees)
        rf_model = MultiOutputRegressor(rf_model_temp, n_jobs = -1)
        rf_model.fit(training_spectra, labels_train_as_array)
        accuracy = rf_model.score(test_spectra, labels_test_as_array)

    if model_type == 'short_dim':
        labels_train = training_labels.drop(columns = ['MiddleDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2', 'log Area/Vol'] )
        labels_test = test_labels.drop(columns = ['MiddleDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2', 'log Area/Vol'] )

        rf_model = RandomForestRegressor(n_estimators = trees, n_jobs = -1, max_features=m_features)
        rf_model.fit(training_spectra, np.ravel(labels_train))
        accuracy = rf_model.score(test_spectra, np.ravel(labels_test))

    if model_type == 'middle_dim':
        labels_train = training_labels.drop(columns = ['ShortestDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2', 'log Area/Vol'] )
        labels_test = test_labels.drop(columns = ['ShortestDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2', 'log Area/Vol'] )

        rf_model = RandomForestRegressor(n_estimators = trees, n_jobs = -1, max_features=m_features)
        rf_model.fit(training_spectra, np.ravel(labels_train))
        accuracy = rf_model.score(test_spectra, np.ravel(labels_test))

    if model_type == 'long_dim':
        labels_train = training_labels.drop(columns=['ShortestDim', 'MiddleDim', 'Geometry_TriangPrismIsosc',
                                                     'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
                                                     'index',
                                                     'Material_Au', 'Material_SiN', 'Material_SiO2', 'log Area/Vol'])
        labels_test = test_labels.drop(columns=['ShortestDim', 'MiddleDim', 'Geometry_TriangPrismIsosc',
                                                'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                'Material_Au', 'Material_SiN', 'Material_SiO2', 'log Area/Vol'])

        rf_model = RandomForestRegressor(n_estimators=trees, n_jobs=-1, max_features=m_features)
        rf_model.fit(training_spectra, np.ravel(labels_train))
        accuracy = rf_model.score(test_spectra, np.ravel(labels_test))

    return (accuracy, rf_model)



def normalize_cm(cm, test_set, num_categories):
    """
    Takes a confusion matrix and normalizes it to show probabilities rather than number of cases
    :param cm: confusion matrix showing numbers of cases rather than probabilities of classification (this is how the
    confusion matrix is outputted in the random forests classifier function)
    :param test_set: the test set used to produce the confusion matrix (also outputted by the RF classifier function)
    NOTE must be a list, not a numpy array, which is what is outputted by the classifier function
    :param num_categories: INT The number of categories in the confusion matrix
    :return: a confusion matrix showing probabilities in each entry rather than number of cases
    """
    normalized_list_cm = []
    for i in range(0, num_categories):
        list_cm = list(cm[i])
        normalized_row_cm = [round(x /test_set.count(i), 2) for x in  list_cm]
        normalized_list_cm.append(normalized_row_cm)
        
    return normalized_list_cm



def plot_accuracy(cm, categories, title, y_range = (0.5,1)):
    """
    plots the accuracy of a model at predicting shapes from spectra, showing the accuracy for each shape

    :param cm: confusion matrix with probabilities at each entry (this is the output of the normalize_cm function)
    :param categories: LIST of STRINGS the shapes predicted by the model (IMPORTANT!!! Make sure the entries in this
    list are in the same order as the diagonal on the confusion matrix)
    :param title: STRING the title of the plot
    :param y_range: LIST of INT/FLOAT range of the y axis for the plot
    :return: Saves the plot as a png file in the same folder as this script. The name will be the title + .png
    """

    accuracies = []
    for i in range(0, len(categories)):
        accuracies.append(cm[i][i])
    sns.barplot(categories, accuracies).set(title = title, ylabel = "Accuracy", ylim = y_range)
    plt.savefig(str(title) + '.png', format='png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def binary_classification_models(training_labels, training_spectra, testing_labels, testing_spectra, binary_model,
                                 confusion_matrix_=True):

    spectra_train_df = pd.DataFrame(training_spectra)
    spectra_test_df = pd.DataFrame(testing_spectra)
    all_data = spectra_train_df.join(training_labels)

    if binary_model == 'Geometry_parallelepiped':
        all_data_shape = all_data.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                'MiddleDim', 'LongDim', 'Material_SiN',
                                                'Material_SiO2', 'Material_Au',
                                                'Geometry_sphere',
                                                'Geometry_wire',
                                                'Geometry_TriangPrismIsosc'])
        labels_test_shape = testing_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                      'MiddleDim', 'LongDim', 'Material_SiN',
                                                      'Material_SiO2', 'Material_Au',
                                                      'Geometry_sphere',
                                                      'Geometry_wire',
                                                      'Geometry_TriangPrismIsosc'])

        all_data_shape_only = all_data_shape[all_data_shape.Geometry_parallelepiped != 0]
        all_data_shape_others = all_data_shape[all_data_shape.Geometry_parallelepiped == 0]
        test = spectra_test_df.join(labels_test_shape)
        labels_test_shape_only = test[test.Geometry_parallelepiped != 0]
        labels_test_shape_others = test[test.Geometry_parallelepiped == 0]

    if binary_model == 'Geometry_sphere':
        all_data_shape = all_data.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                'MiddleDim', 'LongDim', 'Material_SiN',
                                                'Material_SiO2', 'Material_Au',
                                                'Geometry_parallelepiped',
                                                'Geometry_wire',
                                                'Geometry_TriangPrismIsosc'])
        labels_test_shape = testing_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                      'MiddleDim', 'LongDim', 'Material_SiN',
                                                      'Material_SiO2', 'Material_Au',
                                                      'Geometry_parallelepiped',
                                                      'Geometry_wire',
                                                      'Geometry_TriangPrismIsosc'])

        all_data_shape_only = all_data_shape[all_data_shape.Geometry_sphere != 0]
        all_data_shape_others = all_data_shape[all_data_shape.Geometry_sphere == 0]
        test = spectra_test_df.join(labels_test_shape)
        labels_test_shape_only = test[test.Geometry_sphere != 0]
        labels_test_shape_others = test[test.Geometry_sphere == 0]

    if binary_model == 'Geometry_wire':
        all_data_shape = all_data.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                'MiddleDim', 'LongDim', 'Material_SiN',
                                                'Material_SiO2', 'Material_Au',
                                                'Geometry_parallelepiped',
                                                'Geometry_sphere',
                                                'Geometry_TriangPrismIsosc'])
        labels_test_shape = testing_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                      'MiddleDim', 'LongDim', 'Material_SiN',
                                                      'Material_SiO2', 'Material_Au',
                                                      'Geometry_parallelepiped',
                                                      'Geometry_sphere',
                                                      'Geometry_TriangPrismIsosc'])

        all_data_shape_only = all_data_shape[all_data_shape.Geometry_wire != 0]
        all_data_shape_others = all_data_shape[all_data_shape.Geometry_wire == 0]
        test = spectra_test_df.join(labels_test_shape)
        labels_test_shape_only = test[test.Geometry_wire != 0]
        labels_test_shape_others = test[test.Geometry_wire == 0]

    if binary_model == 'Geometry_TriangPrismIsosc':
        all_data_shape = all_data.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                'MiddleDim', 'LongDim', 'Material_SiN',
                                                'Material_SiO2', 'Material_Au',
                                                'Geometry_parallelepiped',
                                                'Geometry_wire',
                                                'Geometry_sphere'])
        labels_test_shape = testing_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                      'MiddleDim', 'LongDim', 'Material_SiN',
                                                      'Material_SiO2', 'Material_Au',
                                                      'Geometry_parallelepiped',
                                                      'Geometry_wire',
                                                      'Geometry_sphere'])

        all_data_shape_only = all_data_shape[all_data_shape.Geometry_TriangPrismIsosc != 0]
        all_data_shape_others = all_data_shape[all_data_shape.Geometry_TriangPrismIsosc == 0]
        test = spectra_test_df.join(labels_test_shape)
        labels_test_shape_only = test[test.Geometry_TriangPrismIsosc != 0]
        labels_test_shape_others = test[test.Geometry_TriangPrismIsosc == 0]

    remove_n = 3176
    drop_indices = np.random.choice(labels_test_shape_others.index, remove_n, replace=False)
    test_subset = labels_test_shape_others.drop(drop_indices)

    test_50_50 = pd.concat([test_subset, labels_test_shape_only])
    shape_binary_test = test_50_50[binary_model]
    test_50_50.drop(columns=[binary_model], inplace=True)

    remove_n = 81205
    drop_indices = np.random.choice(all_data_shape_others.index, remove_n, replace=False)
    df_subset = all_data_shape_others.drop(drop_indices)

    all_data_shape_50_50 = pd.concat([df_subset, all_data_shape_only])
    all_data_shape_50_50.reset_index(inplace=True)
    all_data_shape_50_50.drop(columns=["index"], inplace=True)

    shape_binary = all_data_shape_50_50[binary_model]
    all_data_shape_50_50.drop(columns=[binary_model], inplace=True)

    rf_binary = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf_binary.fit(all_data_shape_50_50, np.ravel(shape_binary))
    rf_binary_accuracy = rf_binary.score(test_50_50, shape_binary_test)
    rf_binary_predictions = rf_binary.predict(test_50_50)
    rf_binary_cm = confusion_matrix(shape_binary_test, rf_binary_predictions)

    cm_normalized = normalize_cm(rf_binary_cm, list(shape_binary_test), 2)
    if confusion_matrix_ == True:
        plot_confusion_matrix(np.asarray(cm_normalized), [binary_model, "Not " + binary_model])
        plt.savefig(binary_model + '.png', format='png')

    return ("Binary Classification Model of " + binary_model, rf_binary, rf_binary_accuracy, cm_normalized,
            rf_binary_predictions, shape_binary_test)

""" Dictionaries used to convert from one hot encoding
from_one_hot_dict = {(1.,0.,0.,0.) : 0, (0.,1.,0.,0.) : 1, (0.,0.,1.,0.) : 2, (0.,0.,0.,1.) : 3}
from_one_hot_dict_materials = {(1.,0.,0.) : 0, (0.,1.,0.) : 1, (0.,0.,1.) : 2}

# Example code to produce shape classification model for only gold and visualize it's accuracy
rf_Au = Train_Random_Forests_Shape_Classification("Au", spectra_train_set, labels_train_set, spectra_test_set, labels_test_set, 50)

accuracy_rf = rf_Au[0]
print(accuracy_rf)
rf_Au_model = rf_Au[1]
cm_Au = rf_Au[2]
rf_Au_test_list = list(rf_Au[4])

cm_Au_normalized = normalize_cm(cm_Au, rf_Au_test_list, 4)
catagories_shape_prediction = ["TriangPrismIsosc", "Parallelepiped", "Sphere", "Wire"]
plot_accuracy(cm_Au_normalized, catagories_shape_prediction, "Accuracy of Shapes Au")

def shape_classification_binary(training_set_spectrum, training_set_lables, test_set_spectrum, test_set_labels,
                                shapes, num_trees = 50, depth = None, m_features = 'auto'):
    for shape in shapes:
        if shape == 'Geometry_parallelepiped':
            labels_train_shape_parallelepiped = training_set_lables.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                           'MiddleDim', 'LongDim', 'Material_SiN',
                                                                           'Material_SiO2', 'Material_Au',
                                                                           'Geometry_sphere',
                                                                           'Geometry_wire',
                                                                           'Geometry_TriangPrismIsosc'])
            labels_test_shape_parallelepiped = test_set_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                         'MiddleDim', 'LongDim', 'Material_SiN',
                                                                         'Material_SiO2', 'Material_Au',
                                                                         'Geometry_sphere',
                                                                         'Geometry_wire', 'Geometry_TriangPrismIsosc'])
            labels_train_shape_parallelepiped_as_array = np.asarray(labels_train_shape_parallelepiped)
            labels_test_shape_parallelepiped_as_array = np.asarray(labels_test_shape_parallelepiped)

            print("Training rf_parallelepiped")
            rf_parallelepiped = RandomForestClassifier(n_estimators = num_trees, max_depth = depth, max_features = m_features, n_jobs=-1)
            rf_parallelepiped.fit(training_set_spectrum, np.ravel(labels_train_shape_parallelepiped_as_array))
            rf_parallelepiped_accuracy = rf_parallelepiped.score(test_set_spectrum,
                                                                 labels_test_shape_parallelepiped_as_array)
            rf_parallelepiped_predictions = rf_parallelepiped.predict(test_set_spectrum)
            rf_parallelepiped_cm = confusion_matrix(labels_test_shape_parallelepiped_as_array,
                                                    rf_parallelepiped_predictions)

        if shape == 'Geometry_TriangPrismIsosc':
            labels_train_shape_triangle = training_set_lables.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                     'MiddleDim', 'LongDim', 'Material_SiN',
                                                                     'Material_SiO2', 'Material_Au', 'Geometry_sphere',
                                                                     'Geometry_wire', 'Geometry_parallelepiped'])
            labels_test_shape_triangle = test_set_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                   'MiddleDim', 'LongDim', 'Material_SiN',
                                                                   'Material_SiO2', 'Material_Au', 'Geometry_sphere',
                                                                   'Geometry_wire', 'Geometry_parallelepiped'])

            labels_train_shape_triangle_as_array = np.asarray(labels_train_shape_triangle)
            labels_test_shape_triangle_as_array = np.asarray(labels_test_shape_triangle)

            print("Training rf_triangle")
            rf_triangle = RandomForestClassifier(n_estimators = num_trees, max_depth = depth, max_features = m_features, n_jobs=-1 )
            rf_triangle.fit(training_set_spectrum, np.ravel(labels_train_shape_triangle_as_array))
            rf_triangle_accuracy = rf_triangle.score(test_set_spectrum, labels_test_shape_triangle_as_array)
            rf_triangle_predictions = rf_triangle.predict(test_set_spectrum)
            rf_triangle_cm = confusion_matrix(labels_test_shape_triangle_as_array, rf_triangle_predictions)

        if shape == 'Geometry_wire':
            labels_train_shape_wire = training_set_lables.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                 'MiddleDim', 'LongDim', 'Material_SiN',
                                                                 'Material_SiO2', 'Material_Au', 'Geometry_sphere',
                                                                 'Geometry_TriangPrismIsosc',
                                                                 'Geometry_parallelepiped'])
            labels_test_shape_wire = test_set_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                               'MiddleDim', 'LongDim', 'Material_SiN',
                                                               'Material_SiO2', 'Material_Au', 'Geometry_sphere',
                                                               'Geometry_TriangPrismIsosc', 'Geometry_parallelepiped'])

            labels_train_shape_wire_as_array = np.asarray(labels_train_shape_wire)
            labels_test_shape_wire_as_array = np.asarray(labels_test_shape_wire)

            print("Training rf_wire")
            rf_wire = RandomForestClassifier(n_estimators = num_trees, max_depth = depth, max_features = m_features, n_jobs=-1)
            rf_wire.fit(training_set_spectrum, np.ravel(labels_train_shape_wire_as_array))
            rf_wire_accuracy = rf_wire.score(test_set_spectrum, labels_test_shape_wire_as_array)
            rf_wire_predictions = rf_wire.predict(test_set_spectrum)
            rf_wire_cm = confusion_matrix(labels_test_shape_wire_as_array, rf_wire_predictions)

        if shape == 'Geometry_sphere':
            labels_train_shape_sphere = training_set_lables.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                   'MiddleDim', 'LongDim', 'Material_SiN',
                                                                   'Material_SiO2', 'Material_Au', 'Geometry_wire',
                                                                   'Geometry_TriangPrismIsosc',
                                                                   'Geometry_parallelepiped'])
            labels_test_shape_sphere = test_set_labels.drop(columns=['index', 'log Area/Vol', 'ShortestDim',
                                                                 'MiddleDim', 'LongDim', 'Material_SiN',
                                                                 'Material_SiO2', 'Material_Au', 'Geometry_wire',
                                                                 'Geometry_TriangPrismIsosc',
                                                                 'Geometry_parallelepiped'])
            labels_train_shape_sphere_as_array = np.asarray(labels_train_shape_sphere)
            labels_test_shape_sphere_as_array = np.asarray(labels_test_shape_sphere)

            print("Training rf_sphere")
            rf_sphere = RandomForestClassifier(n_estimators = num_trees, max_depth = depth, max_features = m_features, n_jobs=-1)
            rf_sphere.fit(training_set_spectrum, np.ravel(labels_train_shape_sphere_as_array))
            rf_sphere_accuracy = rf_sphere.score(test_set_spectrum, labels_test_shape_sphere_as_array)
            rf_sphere_predictions = rf_sphere.predict(test_set_spectrum)
            rf_sphere_cm = confusion_matrix(labels_test_shape_sphere_as_array, rf_sphere_predictions)

    return [["parallelepiped", rf_parallelepiped ,rf_parallelepiped_accuracy, rf_parallelepiped_predictions, rf_parallelepiped_cm,
             labels_test_shape_parallelepiped_as_array],
            ["sphere", rf_sphere, rf_sphere_accuracy, rf_sphere_predictions, rf_sphere_cm, labels_test_shape_sphere_as_array],
            ["triangle", rf_triangle, rf_triangle_accuracy, rf_triangle_predictions, rf_triangle_cm,
             labels_test_shape_triangle_as_array],
            ["wire", rf_wire, rf_wire_accuracy, rf_wire_predictions, rf_wire_cm, labels_test_shape_wire_as_array]]


"""

