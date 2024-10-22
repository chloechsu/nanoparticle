#%% loading ===================================================================
# misc
import sys
import os
from time import strftime,time
from math import pi

# data processing
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat

# ML tools, sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# for saving data
import joblib

#to ignore warnings displayed in notebook
import warnings
warnings.filterwarnings('ignore')

# homemade functions
from data_gen_utils import gen_data_P1_P2_P3_Elzouka
from io_utils import load_spectrum_param_data_mat, split_and_write_to_csv
from eval_utils import calc_RMSE_MAE_MSE_Erel, z_RF_DT_DTGEN_error_folds, spectra_prediction_corrector

#%% inputs ====================================================================
n_estimators = 200
test_size = 0.2
n_cpus = 8
num_folds_training_for_errors = 2 # 100
n_gen_to_data_ratio = 20 # the ratio between n_gen to the data used for ML

train_datasize_fraction_scalar = 0.5 # the fraction of original data to be used for ML.
train_datasize_fraction_spectral = 0.5 # the fraction of original data to be used for ML.
n_data_desired = {'Geometry_sphere': 500, 'Geometry_wire': 800, 'Geometry_parallelepiped': 2000, 'Geometry_TriangPrismIsosc': 2000}

use_log_emissivity = True # True: use log emissivity as input to ML, this will make the training target to monimize relative error (i.e., MINIMIZE( log(y_pred) - log(y_test) ) is equivalent to MINIMIZE( log(y_pred / y_test) ))

data_size_fraction_ = np.array([0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
data_size_fraction_ = np.array([0.5])
Models_to_Study_performanceVSsize = ['DTGEN', 'DT']                    
#data_size_fraction_ = np.array([0.01,0.02,0.05])
num_folds_repeat_DataReduction=1

#spectral_or_scalar_calc_all = ['scalar','spectral'] # list that contains either 'scalar' or 'spectral' or both
spectral_or_scalar_calc_all = ['spectral'] # list that contains either 'scalar' or 'spectral' or both

datetime_str = strftime("%Y%m%d_%H%M%S")

matlab_data_path = 'data/simulated_data.mat'

optional_title_folders='{0}sc_{1}sp_{2}_CPU'.format(train_datasize_fraction_scalar*100, train_datasize_fraction_spectral*100, n_cpus)
#optional_title_folders='DataSize_study_BigTest'

feature_set = ['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
               'Material_Au', 'Material_SiN', 'Material_SiO2',
               'log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim']

#feature_set = ['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
#               'Material_Au', 'Material_SiN', 'Material_SiO2',
#               'log Area/Vol', 'log ShortestDim', 'log MiddleDim', 'log LongDim']

# DON'T CHANGE THESE VARIABLES
my_x = np.logspace(13,14.9,num=400)
scaling_factors = {'Area':10.0**12,'Volume':10.0**18,'Volume/Area':10.0**9,'Area/Volume':10.0**-6,'LongestDim':10.0**6,'MiddleDim':10.0**6,'ShortestDim':10.0**6,'PeakFrequency':10.0**-14,'PeakEmissivity':10.0,'LengthX':10**6,'Height':10**6,'Length':10**6}


#%% Loading data ==============================================================
data_featurized,interpolated_ys,spectrum_parameters = load_spectrum_param_data_mat(loadmat(matlab_data_path)['All_data'].T, my_x, scaling_factors) # this data has more "sphere" results

#%% Data pre-processing =======================================================

# separating material and geometry features, for stratifying the data
feature_set_geom = [x for x in feature_set if 'Geometry' in x ]
feature_set_mat = [x for x in feature_set if 'Material' in x ]
feature_set_geom_mat = feature_set_geom + feature_set_mat
feature_set_dimensions = list(set(feature_set) - set(feature_set_geom_mat))

print("dataset size, original: {0}".format(len(data_featurized)))

# drop data with A/V outside the range
sclae_here = scaling_factors['Area/Volume']
idx_low  = data_featurized['Area/Vol'].values<0.9e6*sclae_here
idx_high = data_featurized['Area/Vol'].values>1e8*sclae_here
idx_outrange = ~np.logical_or(idx_low, idx_high)
data_featurized = data_featurized[idx_outrange]
interpolated_ys = interpolated_ys[idx_outrange]
print("after dropping A/V outside range, we have {0} datapoints".format(len(data_featurized)))

#drop analytical (except spheres) samples from training
idx_num = ~data_featurized['is_analytical'].astype(bool)
idx_anal = ~idx_num
print("we have {0} numerical simulations, and {1} analytical".format(sum(idx_num), sum(idx_anal)))

idx_sphere = data_featurized['Geometry_sphere'].astype(bool)
print("we have {0} spheres".format(sum(idx_sphere)))

# we will keep all numerical simulations or sphere simulations
idx = np.logical_or(idx_num.values, idx_sphere.values)
print("we have {0} spheres | numerical simulation".format(sum(idx)))
data_featurized = data_featurized[idx]
interpolated_ys = interpolated_ys[idx]


# to reduce the number of datapoints for each geometry
#idx_to_keep = []
idx_to_keep = np.array([])
for geom in feature_set_geom:
    n_data_desired_here = n_data_desired[geom]
    idx_geom_here = data_featurized[geom]
    for mat in feature_set_mat:
        idx_mat_here = data_featurized[mat]
        #idx_geom_mat_here = np.logical_and(idx_geom_here.values, idx_mat_here.values)
        idx_geom_mat_here = np.where((data_featurized[geom].values == 1) & (data_featurized[mat].values == 1))[0]
        if n_data_desired_here >= len(idx_geom_mat_here):
            #idx_to_keep.append(idx_geom_mat_here)
            idx_to_keep = np.append(idx_to_keep, idx_geom_mat_here)            
        else:
            #idx_to_keep.append(np.random.choice(idx_geom_mat_here, size=n_data_desired_here, replace=False) )
            idx_to_keep = np.append(idx_to_keep, np.random.choice(idx_geom_mat_here, size=n_data_desired_here, replace=False) ) 
        
       
data_featurized = data_featurized.iloc[idx_to_keep]
interpolated_ys = interpolated_ys[idx_to_keep.astype(int)]

assert len(interpolated_ys) == len(data_featurized)

data_featurized = data_featurized.drop('is_analytical',axis=1)

print("dataset size, after dropping: {0}".format(len(data_featurized)))


#%% inference RF and DT. Create DTGEN =========================================
time_train = {}
time_pred = {}
for spectral_or_scalar_calc in spectral_or_scalar_calc_all:
    # folder to save the data
    save_folder = 'cache/r{0}_'.format(datetime_str) + optional_title_folders + '/' + spectral_or_scalar_calc + '/'
    os.makedirs(save_folder, exist_ok=True)
    
    # preparing features and labels
    X = data_featurized[feature_set].copy()    
    if spectral_or_scalar_calc == 'spectral':
        y = interpolated_ys
        train_data_size_fraction = train_datasize_fraction_spectral
    elif spectral_or_scalar_calc == 'scalar':
        y = data_featurized['Emissivity'].values
        train_data_size_fraction = train_datasize_fraction_scalar

    # to reduce datasize
    test_size = 1-train_data_size_fraction
    
    # making sure that our X is float
    X = X.astype(np.float64)    
    
    # analysing RF and DT error, if trained multiple times
    for RF_DT__ in ['RF', 'DT']:
        z_RF_DT_DTGEN_error_folds(X, y, feature_set, feature_set_dimensions,
                feature_set_geom_mat, data_featurized, my_x,
                num_folds=num_folds_training_for_errors, test_size=test_size,
                n_estimators=n_estimators, n_cpus=n_cpus, keep_spheres=True,
                optional_title_folders=save_folder,
                use_log_emissivity=use_log_emissivity,
                display_txt_out=True, RF_or_DT__=[RF_DT__],
                PlotTitle_extra=spectral_or_scalar_calc)
    
    # splitting test and training
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,stratify=X[feature_set_geom_mat])    
    if use_log_emissivity:
        y_train = np.log(y_train)
        y_train[y_train<-25] = -25 # if we have 'ZEROS', its log will be -Inf. We are replacing anything zlose to zero with exp(-25)
        
    
    print("Features used: {0}".format(X.columns))
    print("We have {0} training and {1} test samples here".format(len(X_train), len(X_test)))
    n_train = len(X_train)        
        
    
    ## training and testing RF
    print("\nRF inference, {0} =========================================================".format(spectral_or_scalar_calc))
    rf = RandomForestRegressor(n_estimators,n_jobs=n_cpus)
    
    start_time = time()
    rf.fit(X_train,y_train)
    time_train['RF_'+spectral_or_scalar_calc] = time() - start_time
    
    start_time = time()
    if use_log_emissivity:
        y_pred_rf = np.exp(rf.predict(X_test))    
    else:
        y_pred_rf = spectra_prediction_corrector(rf.predict(X_test))      
    time_pred['RF_'+spectral_or_scalar_calc] = time() - start_time    
    
    print("RF error analysis")
    rf_r2,rf_mae,rf_mse,rf_Erel, rf_r2_all,rf_mae_all,rf_mse_all,rf_Erel_all = calc_RMSE_MAE_MSE_Erel(y_test,y_pred_rf, my_x)    
    
    
    ## training and testing DT
    print("\nDT inference, {0} =========================================================".format(spectral_or_scalar_calc))
    dt = DecisionTreeRegressor()
    
    start_time = time()
    dt.fit(X_train,y_train)
    time_train['DT_'+spectral_or_scalar_calc] = time() - start_time
    
    start_time = time()
    if use_log_emissivity:
        y_pred_dt = np.exp(dt.predict(X_test))    
    else:
        y_pred_dt = spectra_prediction_corrector(dt.predict(X_test))
    time_pred['DT_'+spectral_or_scalar_calc] = time() - start_time    
        
    print("DT error analysis")    
    dt_r2,dt_mae,dt_mse,dt_Erel, dt_r2_all,dt_mae_all,dt_mse_all,dt_Erel_all = calc_RMSE_MAE_MSE_Erel(y_test,y_pred_dt, my_x)    
    
    
    ## Generating data for DTGEN, training and testing DTGEN
    print("\nDT GEN inference, {0} =====================================================".format(spectral_or_scalar_calc))    
    X_train_all_columns = data_featurized.loc[X_train.index,:]
    start_time = time()
    n_gen = int(len(X_train_all_columns) * n_gen_to_data_ratio)

    X_train_all_materials = X_train
    y_train_all_materials = y_train
    X_test_all_materials = X_test
    y_test_all_materials = y_test

    for material in ['Au', 'SiN', 'SiO2']:
        X_gen = gen_data_P1_P2_P3_Elzouka(X_train_all_columns,n_gen,
                material=material)
        X_gen = pd.DataFrame(X_gen,columns=X_train.columns).astype(np.float64)
        X_gen = X_gen[feature_set]
        end_time = time() 
        print('done generating input features for DTGEN in {0} seconds'.format(end_time-start_time))
        time_DTGEN_feature_creation = end_time - start_time
        
        # predicting emissivity using RF for the generated data ------------------
        start_time = time()
        if use_log_emissivity:    
            y_gen = np.exp(rf.predict(X_gen))  
        else:
            y_gen = spectra_prediction_corrector(rf.predict(X_gen))
        end_time = time() 
        print('done predicting emissivity using the input features using RF in {0} seconds'.format(end_time-start_time))
        time_DTGEN_label_creation = end_time - start_time
    
        if material is None:
            material = 'all'
            X_train, y_train = X_train_all_materials, y_train_all_materials
            X_test, y_test = X_test_all_materials, y_test_all_materials
        else:
            mask = (X_train_all_materials['Material_' + material] == 1)
            X_train, y_train = X_train_all_materials[mask], y_train_all_materials[mask] 
            mask = (X_test_all_materials['Material_' + material] == 1)
            X_test, y_test = X_test_all_materials[mask], y_test_all_materials[mask] 
        # adding the generated emissivity to original training emissivity ------------------
        if use_log_emissivity:
            X_new_train,y_new_train = pd.concat([X_gen,X_train]),np.concatenate([np.log(y_gen),y_train])        
        else:
            X_new_train,y_new_train = pd.concat([X_gen,X_train]),np.concatenate([y_gen,y_train])

        # creating a single decision tree trained on generated and original training emissivity
        new_n_train = n_gen + n_train
        dt_gen = DecisionTreeRegressor(min_samples_leaf=3)
        
        start_time = time()
        dt_gen.fit(X_new_train,y_new_train)
        time_train['DTGEN_'+spectral_or_scalar_calc] = time() - start_time
        
        start_time = time()
        if use_log_emissivity:
            y_pred_dtgen    = np.exp(dt_gen.predict(X_test))
            y_new_train     = np.exp(y_new_train)
            y_train         = np.exp(y_train)
        else:
            y_pred_dtgen = dt_gen.predict(X_test)
        time_pred['DTGEN_'+spectral_or_scalar_calc] = time() - start_time        
        
        print("DTGEN error analysis")
        dt_gen_r2,dt_gen_mae,dt_gen_mse,dt_gen_Erel, dt_gen_r2_all,dt_gen_mae_all,dt_gen_mse_all,dt_gen_Erel_all = calc_RMSE_MAE_MSE_Erel(y_test,y_pred_dtgen, my_x)    

        # split_and_write_to_csv(X_train, 'data/sim_train_labels_%s.csv' %
        #         material)
        # split_and_write_to_csv(y_train, 'data/sim_train_spectrum_%s.csv' %
        #         material)
        split_and_write_to_csv(X_gen, 'data/gen_labels_%s.csv' %
                material)
        split_and_write_to_csv(y_gen, 'data/gen_spectrum_%s.csv' %
                material)
        # Split test into half validation and half test.
        # X_val, X_test_split, y_val, y_test_split = train_test_split(X_test, y_test,
        #         test_size=0.5, stratify=X_test[feature_set_geom_mat])
        # split_and_write_to_csv(X_val,
        #         'data/sim_validation_labels_%s.csv' % material)
        # split_and_write_to_csv(y_val,
        #         'data/sim_validation_spectrum_%s.csv' % material)
        # split_and_write_to_csv(X_test_split,
        #         'data/sim_test_labels_%s.csv' % material)
        # split_and_write_to_csv(y_test_split,
        #         'data/sim_test_spectrum_%s.csv' % material)
        
        
        #%% save ML models, test and train data ===================================
        # Save in Python format
        # variable_name_list = ['rf', 'dt', 'dt_gen',                           
        #                       'X_train', 'X_new_train', 'X_test', 
        #                       'y_train', 'y_new_train', 'y_test',
        #                       'n_gen', 'train_data_size_fraction', 'my_x', 'scaling_factors']
        # for variable_name in variable_name_list:
        #     joblib.dump(globals()[variable_name],
        #             save_folder+variable_name+'_'+material+'.joblib')        
