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



spectra_train_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/y_new_train.joblib')
spectra_test_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/y_test.joblib')
labels_train_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/x_new_train.joblib').reset_index()
labels_test_set = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/x_test.joblib').reset_index()
labels_train_smaller = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/x_train.joblib').reset_index()
spectra_train_smaller = joblib.load('cache/r20200406_234541_50.0sc_50.0sp_1_CPU/spectral/y_train.joblib')



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



from_one_hot_dict = {(1.,0.,0.,0.) : 0, (0.,1.,0.,0.) : 1, (0.,0.,1.,0.) : 2, (0.,0.,0.,1.) : 3}
from_one_hot_dict_materials = {(1.,0.,0.) : 0, (0.,1.,0.) : 1, (0.,0.,1.) : 2}



def convert_from_one_hot(df_as_array, dictionary):
    catagories_list = []
    for row in df_as_array:
        row_tuple = tuple(row)
        catagories_list.append(dictionary[row_tuple])
    catagories_list_array = np.asarray(catagories_list)
    print('done')
    return catagories_list_array


def Train_Random_Forests_Shape_Classification(model_type, training_spectra, training_labels, 
                                              test_spectra, test_labels, trees):
    if model_type == 'All':
        labels_train_shape = training_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim', 
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2', 'index'] )
        labels_test_shape = test_labels.drop(columns = ['index','log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim', 
                                                        'Material_Au','Material_SiN', 'Material_SiO2', 'index'] )

        labels_train_shape_as_array = labels_train_shape.to_numpy()
        labels_test_shape_as_array = labels_test_shape.to_numpy()
        
        labels_train_shape_as_array_wo_OHE = convert_from_one_hot(labels_train_shape_as_array, from_one_hot_dict)
        labels_test_shape_as_array_wo_OHE = convert_from_one_hot(labels_test_shape_as_array, from_one_hot_dict)
    
        rf_model = RandomForestClassifier(n_estimators = trees)
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
        
    if model_type == 'Au' or 'SiN' or 'SiO2':
        
        spectra_train_df = pd.DataFrame(training_spectra)
        spectra_test_df = pd.DataFrame(test_spectra)

        spectra_train_df.drop(indicies_to_drop_train_list, inplace=True)
        spectra_test_df.drop(indicies_to_drop_test_list, inplace=True)

        labels_train_shape_as_array = labels_train_shape.to_numpy()
        labels_test_shape_as_array = labels_test_shape.to_numpy()

        labels_train_shape_as_array_wo_OHE = convert_from_one_hot(labels_train_shape_as_array, from_one_hot_dict)
        labels_test_shape_as_array_wo_OHE =  convert_from_one_hot(labels_test_shape_as_array, from_one_hot_dict)

        spectra_train_shape_as_array = spectra_train_df.to_numpy()
        spectra_test_shape_as_array= spectra_test_df.to_numpy()
        
        rf_model = RandomForestClassifier(n_estimators = trees)
        rf_model.fit(spectra_train_shape_as_array, labels_train_shape_as_array_wo_OHE)
        accuracy = rf_model.score(spectra_test_shape_as_array, labels_test_shape_as_array_wo_OHE)
        
        predictions = rf_model.predict(spectra_test_shape_as_array)
        cm_rf = confusion_matrix(labels_test_shape_as_array_wo_OHE, predictions)
    
    return (accuracy, rf_model, cm_rf, predictions, labels_test_shape_as_array_wo_OHE)



def Train_Random_Forests_Size_Regression(model_type, training_spectra, training_labels, 
                                              test_spectra, test_labels, trees):
    if model_type == 'volume':
        labels_train = training_labels.drop(columns = ['ShortestDim', 'MiddleDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                            'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        labels_test = test_labels.drop(columns = ['ShortestDim', 'MiddleDim', 'LongDim', 'Geometry_TriangPrismIsosc',
                                                            'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire', 'index',
                                                             'Material_Au', 'Material_SiN', 'Material_SiO2'] )
        rf_model = RandomForestRegressor(n_estimators = trees, n_jobs = -1)
        rf_model.fit(training_spectra, np.ravel(labels_train))
        accuracy = rf_model.score(test_spectra, np.ravel(labels_test))
    
    if model_type == 'All':
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
        
    
    
    return (accuracy, rf_model)



def normalize_cm(cm, test_set, num_catagories):
    normalized_list_cm = []
    for i in range(0, num_catagories):
        list_cm = list(cm[i])
        normalized_row_cm = [x /test_set.count(i) for x in  list_cm]
        normalized_list_cm.append(normalized_row_cm)
        
    return normalized_list_cm



def plot_accuracy(cm, catagories, title, y_range = [0.5,1]):
    accuracies = []
    for i in range(0, len(catagories)):
        accuracies.append(cm[i][i])
    sns.barplot(catagories, accuracies).set(title = title, ylabel = "Accuracy", ylim = y_range)
    plt.savefig(str(title) + '.png', format='png')



# Example code to produce shape classification model for only gold
rf_Au = Train_Random_Forests_Shape_Classification("Au", spectra_train_set, labels_train_set, spectra_test_set, labels_test_set, 50)

accuracy_rf = rf_Au[0]
print(accuracy_rf)
rf_Au_model = rf_Au[1]
cm_Au = rf_Au[2]
rf_Au_test_list = list(rf_Au[4])

cm_Au_normalized = normalize_cm(cm_Au, rf_Au_test_list, 4)
catagories_shape_prediction = ["TriangPrismIsosc", "Parallelepiped", "Sphere", "Wire"]
plot_accuracy(cm_Au_normalized, catagories_shape_prediction, "Accuracy of Shapes Au")




