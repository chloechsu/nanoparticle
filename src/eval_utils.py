import os
from time import time

import numpy as np
import pandas as pd
import scipy

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from data_gen_utils import gen_data_P1_P2_P3_Elzouka


#%% ERROR ANALYSIS ==========================================================================
# calculating relative error for spectrum
def error_integ_by_spectrum_integ(y_test, y_pred, x = []):
    """This function calculates relative error for either a spectrum or scalar target
    The result is the ratio between:
        the absolute error between test&pred, integrated by np.trapz
        ------------------------------------------------------------
        the value of y, integrated by np.trapz
        
        if the input is 1D array, the integral is omitted
        """
    if len(y_test.shape) == 2: # if y_test is 2D "i.e., spectral emissivity"
        if len(x) == 0:
            y_test_integ = np.trapz(y_test, axis=1)
        else:
            y_test_integ = np.trapz(y_test, x, axis=1)                
    else: # if y_test is 1D "i.e., scalar emissivity"
        y_test_integ = y_test
    
    error_abs = np.abs(y_test - y_pred)
    
    if len(y_test.shape) == 2: # if y_test is 2D "i.e., spectral emissivity"
        if len(x) == 0:
            error_abs_integ = np.trapz(error_abs, axis=1)
        else:
            error_abs_integ = np.trapz(error_abs, x, axis=1)
    else: # if y_test is 1D "i.e., scalar emissivity"
        error_abs_integ = error_abs    
    
    error_rel_integ = error_abs_integ/y_test_integ
    
    return error_rel_integ,np.mean(error_rel_integ)


def RMSE(y_actual,y_pred):
    return np.sqrt(mean_squared_error(y_actual,y_pred))


def calc_RMSE_MAE_MSE_Erel(y_test,y_pred, my_x, printing=True):
    """
    calculate the errors; averaged and for all test-pred elements
    my_x: the frequency points, required only for spectral emissivity
    """
    
    # error metrics, averaged
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    Erel_all,Erel = error_integ_by_spectrum_integ(y_test,y_pred, my_x)
    
    # error metrics, for all elements of test-pred
    r2_all = r2_score(y_test,y_pred, multioutput='raw_values')
    mae_all = mean_absolute_error(y_test,y_pred, multioutput='raw_values')
    mse_all = mean_squared_error(y_test,y_pred, multioutput='raw_values')
    
    if printing:
        print("R2: {0:.8f}".format(r2))
        print("MAE: {0:.8f}".format(mae))
        print("MSE: {0:.8f}".format(mse))
        print("Erel: {0:.8f}".format(Erel))
    
    return r2,mae,mse,Erel, r2_all,mae_all,mse_all, Erel_all 


def spectra_prediction_corrector(y):
    """To replace any negative emissivity values with 'ZEROS'"""
    assert isinstance(y,np.ndarray)
    y_copy = np.copy(y)
    y_copy[y_copy<0] = 0
    return y_copy


def z_RF_DT_DTGEN_error_folds(X_reduced,y_reduced, feature_set, feature_set_dimensions, feature_set_geom_mat, data_featurized, my_x, \
                     num_folds=20, test_size=0.2, n_estimators=200, n_cpus = 1, keep_spheres = True, optional_title_folders='', \
                     use_log_emissivity=True, display_txt_out = True, RF_or_DT__ = ['RF'], PlotTitle_extra = '', \
                     n_gen_to_data_ratio=150):
    '''
    INPUTS that is required only for DTGEN
    data_featurized: all the data, with all the columns. Required only for DTGEN
    n_gen_to_data_ratio : ratio between the amont of data generated for DTGEN and the training data
    '''
    
    #determine_spectral_or_scalar
    if len(y_reduced.shape) == 2: # if y is 2D "i.e., spectral emissivity" 
        spectral_or_scalar_calc = 'spectral'
    else: # if y is 1D "i.e., scalar emissivity"
        spectral_or_scalar_calc = 'scalar'
        
    
    index_data_here = np.array(X_reduced.index)
            

    ##get errors w.r.t material/geometry intersectionality, also get runtime
    mae,rmse,r2,mse,Erel = [],[],[],[],[]
    mae_matgeom,rmse_matgeom,r2_matgeom,mse_matgeom,Erel_matgeom,ntest_matgeom,ntrain_matgeom={},{},{},{},{},{},{}
    mats = ['Material_SiO2','Material_Au','Material_SiN']
    mats_colors = ['b','r','m']
    geoms = ['Geometry_TriangPrismIsosc','Geometry_parallelepiped','Geometry_wire','Geometry_sphere']    
    train_time_feat,pred_time_feat = [],[]

    metric_list = ['mae', 'r2', 'mse', 'rmse', 'Erel', 'ntest', 'ntrain']
  
    All_errors = {}
    pred_time = {}
    train_time = {}
    for predictor in ['RF', 'DT', 'DTGEN'] :
        pred_time[predictor] = []
        train_time[predictor] = []
        All_errors[predictor] = {} 
        for metric in metric_list:
            All_errors[predictor][metric + '_matgeom'] = {}
    
    
    for i in range(num_folds):            
        X_train,X_test,y_train,y_test = train_test_split(X_reduced,y_reduced,test_size=test_size,stratify=X_reduced[feature_set_geom_mat])
        
        if use_log_emissivity:
            y_train = np.log(y_train)
            y_train[y_train<-25] = -25
        
        for RF_or_DT in RF_or_DT__:
            print('Analyzing the error for {0}, running training for {1}th time out of {2} times =========='.format(RF_or_DT, i, num_folds))
            
            if RF_or_DT == 'RF' or RF_or_DT == 'DTGEN':
                estimator = RandomForestRegressor(n_estimators,n_jobs=n_cpus, criterion='mse')
                estimator_type='RF'
            elif RF_or_DT == 'DT':
                estimator = DecisionTreeRegressor()
                estimator_type='DT'
            
            start_time = time()
            estimator.fit(X_train,y_train)
            end_time = time()        
            train_time_feat.append((10**6*(end_time-start_time))/float(len(X_train)))
            train_time[estimator_type].append((end_time-start_time))
    
            start_time = time()
            if use_log_emissivity:
                y_pred = np.exp(estimator.predict(X_test))                
            else:
                y_pred = spectra_prediction_corrector(estimator.predict(X_test))
            end_time = time()
            pred_time_feat.append((10**6*(end_time-start_time))/float(len(X_test)))
            pred_time[estimator_type].append((end_time-start_time))
    
            r2_here = r2_score(y_test,y_pred)
            mae_here = mean_absolute_error(y_test,y_pred)
            mse_here = mean_squared_error(y_test,y_pred)
            rmse_here = RMSE(y_test,y_pred)
            _,Erel_here = error_integ_by_spectrum_integ(y_test, y_pred, my_x)
    
            r2.append(r2_here)
            mae.append(mae_here)
            mse.append(mse_here)        
            rmse.append(rmse_here)
            Erel.append(Erel_here)

            error_dict = All_errors[estimator_type]
            
            # errors broken by material and geometry
            for m in mats:
                for g in geoms:
                    formal_material = m.split('_')[1]
                    formal_geom = g.split('_')[1]
                    key_str = formal_material + ' ' + formal_geom
                    idx = (X_test[m]==1)&(X_test[g]==1)
                    ntest = sum(idx) # number of test data
                    idx_train = (X_train[m]==1)&(X_train[g]==1)
                    ntrain = sum(idx_train) # number of training data
                    if i == 0:
                        for metric in metric_list:
                            error_dict[metric+'_matgeom'][key_str] = []
    
                    if sum(idx)!=0:
                        error_dict['mae_matgeom'][key_str].append(
                                mean_absolute_error(y_test[idx],y_pred[idx]))
                        error_dict['r2_matgeom'][key_str].append(
                                r2_score(y_test[idx],y_pred[idx]))
                        error_dict['mse_matgeom'][key_str].append(
                                mean_squared_error(y_test[idx],y_pred[idx]))
                        error_dict['rmse_matgeom'][key_str].append(
                                RMSE(y_test[idx],y_pred[idx]))
                        _, Erel_here = error_integ_by_spectrum_integ(y_test[idx], y_pred[idx], my_x)
                        error_dict['Erel_matgeom'][key_str].append(Erel_here)
                        error_dict['ntest_matgeom'][key_str].append(ntest)
                        error_dict['ntrain_matgeom'][key_str].append(ntrain)
                        
            # DTGEN
            if RF_or_DT == 'DTGEN' and estimator_type == 'RF':
                start_time_DTGEN_train = time()
                X_train_all_columns = data_featurized.loc[X_train.index,:]
                start_time = time()
                n_gen = int(len(X_train_all_columns) * n_gen_to_data_ratio)
                X_gen = gen_data_P1_P2_P3_Elzouka(X_train_all_columns,n_gen);
                X_gen = pd.DataFrame(X_gen,columns=X_train.columns).astype(np.float64)
                X_gen = X_gen[feature_set]
                end_time = time() 
                print('done generating input features for DTGEN in {0} seconds'.format(end_time-start_time))
                time_DTGEN_feature_creation = start_time - end_time
                
                # predicting emissivity using RF for the generated data ------------------
                start_time = time()
                if use_log_emissivity:    
                    y_gen = np.exp(estimator.predict(X_gen))  
                else:
                    y_gen = spectra_prediction_corrector(estimator.predict(X_gen))
                end_time = time() 
                print('done predicting emissivity using the input features using RF in {0} seconds'.format(end_time-start_time))
                time_DTGEN_label_creation = start_time - end_time
                
                # adding the generated emissivity to original training emissivity ------------------
                if use_log_emissivity:
                    X_new_train,y_new_train = pd.concat([X_gen,X_train]),np.concatenate([np.log(y_gen),y_train])        
                else:
                    X_new_train,y_new_train = pd.concat([X_gen,X_train]),np.concatenate([y_gen,y_train])
                
                # creating a single decision tree trained on generated and original training emissivity            
                dt_gen = DecisionTreeRegressor(min_samples_leaf=3)
                
                start_time = time()
                dt_gen.fit(X_new_train,y_new_train)
                end_time_DTGEN_train = time()
                train_time['DTGEN'].append((end_time_DTGEN_train-start_time_DTGEN_train))
                
                start_time = time()
                if use_log_emissivity:
                    y_pred_dtgen    = np.exp(dt_gen.predict(X_test))
                    y_new_train     = np.exp(y_new_train)                    
                else:
                    y_pred_dtgen = dt_gen.predict(X_test) 
                end_time = time()
                pred_time['DTGEN'].append((end_time-start_time))
                
                #print("DTGEN error analysis")
                #dt_gen_r2,dt_gen_mae,dt_gen_mse,dt_gen_Erel, dt_gen_r2_all,dt_gen_mae_all,dt_gen_mse_all,dt_gen_Erel_all = calc_RMSE_MAE_MSE_Erel(y_test,y_pred_dtgen, my_x)
                
                y_pred = y_pred_dtgen
                # errors broken by material and geometry
                estimator_type = 'DTGEN'
                for m in mats:
                    for g in geoms:
                        formal_material = m.split('_')[1]
                        formal_geom = g.split('_')[1]
                        idx = (X_test[m]==1)&(X_test[g]==1) ; ntest = sum(idx) # number of test data
                        idx_train = (X_train[m]==1)&(X_train[g]==1) ; ntrain = sum(idx_train) # number of training data
                        if i == 0:
                            All_errors[estimator_type]['mae_matgeom'][formal_material+' '+formal_geom] = []
                            All_errors[estimator_type]['r2_matgeom'][formal_material+' '+formal_geom] = []
                            All_errors[estimator_type]['mse_matgeom'][formal_material+' '+formal_geom] = []
                            All_errors[estimator_type]['rmse_matgeom'][formal_material+' '+formal_geom] =[]
                            All_errors[estimator_type]['Erel_matgeom'][formal_material+' '+formal_geom] =[]
                            All_errors[estimator_type]['ntest_matgeom'][formal_material+' '+formal_geom] = []
                            All_errors[estimator_type]['ntrain_matgeom'][formal_material+' '+formal_geom] =[]
        
                        if sum(idx)!=0:
                            All_errors[estimator_type]['mae_matgeom'][formal_material+' '+formal_geom].append(mean_absolute_error(y_test[idx],y_pred[idx]))
                            All_errors[estimator_type]['r2_matgeom'][formal_material+' '+formal_geom].append(r2_score(y_test[idx],y_pred[idx]))
                            All_errors[estimator_type]['mse_matgeom'][formal_material+' '+formal_geom].append(mean_squared_error(y_test[idx],y_pred[idx]))
                            All_errors[estimator_type]['rmse_matgeom'][formal_material+' '+formal_geom].append(RMSE(y_test[idx],y_pred[idx]))
                            _,Erel_here=error_integ_by_spectrum_integ(y_test[idx], y_pred[idx], my_x)
                            All_errors[estimator_type]['Erel_matgeom'][formal_material+' '+formal_geom].append(Erel_here)
                            All_errors[estimator_type]['ntest_matgeom'][formal_material+' '+formal_geom].append(ntest)
                            All_errors[estimator_type]['ntrain_matgeom'][formal_material+' '+formal_geom].append(ntrain)
            
        

    ## saving the errors
    ############# Saving the data #####################################
    savefolder = optional_title_folders+'/'
    filename_mat = 'inference_'+spectral_or_scalar_calc+'_'+RF_or_DT+'_errors_averaged_over_{0}_runs'.format(num_folds)    
    os.makedirs(savefolder, exist_ok=True)

    # Save in Matlab format
    dict_to_save = {}
    #variable_name_list = ['mae','mse','r2','rmse','Erel','mae_matgeom','rmse_matgeom','r2_matgeom','mse_matgeom','Erel_matgeom','matlab_data_path', 'feature_set', 'num_folds', 'index_data_here', 'ntrain_matgeom', 'ntest_matgeom']        
    variable_name_list = ['All_errors', 'train_time', 'pred_time']
    for variable_name in variable_name_list:
        if variable_name in locals():
            dict_to_save[variable_name] = locals()[variable_name] # here, we use "LOCALS()", rather than "GLOBALS()"
        elif variable_name in globals():
            dict_to_save[variable_name] = globals()[variable_name] # here, we use "LOCALS()", rather than "GLOBALS()"
        else:
            print("{0} does not exist in local or global variables".format(variable_name))
    scipy.io.savemat(savefolder+filename_mat+'.mat', dict_to_save)

    if display_txt_out:
        print(feature_set_dimensions)
        print('R^2: {0:.6f} pm {1:.6f}'.format(np.average(r2),2*np.std(r2)))
        print('MAE: {0:.6f} pm {1:.6f}'.format(np.average(mae),2*np.std(mae)))
        print('RMSE: {0:.6f} pm {1:.6f}'.format(np.average(rmse),2*np.std(rmse)))
        print('MSE: {0:.6f} pm {1:.6f}'.format(np.average(mse),2*np.std(mse)))
        print('Erel: {0:.6f} pm {1:.6f}'.format(np.average(Erel),2*np.std(Erel)))

        print("average runtime per sample on lawrencium (clock seconds): {0}".format(data_featurized[data_featurized["Runtime"]>0]['Runtime'].mean()*3600))
        print("Average runtime per sample for "+RF_or_DT+" training feature(CPU seconds): {0}".format(np.average(train_time_feat)*n_cpus/10**6))
        print("Average runtime per sample for "+RF_or_DT+" inference feature prediction(CPU seconds): {0}".format(np.average(pred_time_feat)*n_cpus/10**6))
