# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 08:00:32 2019

@author: local-admin
"""
#%% Importing libraries and modules, and loading data
#%% loading ===================================================================
#misc
import sys
import os
from time import strftime,time
from joblib import dump, load

#data processing
import numpy as np
import pandas as pd

# for saving data
import joblib
import glob

from inv_model_utils import DoInverseDesign_SaveCSV, draw_target_spectrum_from_low_high_ranges

#to ignore warnings
import warnings
warnings.filterwarnings('ignore')


#%% Inputs
date_for_dirname = 'd_'+strftime("%Y%m%d_%H%M%S") + '_'

use_log_emissivity = True
DT_or_DTGEN__ = ['DT', 'DTGEN'] #DT_or_DTGEN__ = ['DT'] #

gen_n = 2500 # number of posible InverseDesign features suggested by ML model

InverseDesign_scalar = True
target_scalar_emissivity_ = [0.1, 0.5, 2, 5]

InverseDesign_spectral = True

RadCooling = False

ChooseRandomFromTest_eachMat = False;

IndexTest_targets = False
idx_Test_targets = [1386] # index in Test that we need to target for the inverse design

random_multipeak = False

peak_at_lambda = False
target_peak_center_um = 7.5
target_peak_width_rads = 6e13

bestdesign_method =  'MorethanOneMatGeom' #'absolute_best'

#InverseDesignModels_folder = 'latest' # here, input the string 'latest' to use
#the latest models, or input the folder of the latest model relative to the
#current folder
InverseDesignModels_folder = 'latest'

#in case we need to display design rules for a number of leaves. If we really need a neat inverse design, set this to -1
n_best_leaves_force=10


# defining the InverseDesign folder
if InverseDesignModels_folder == 'latest':
    all_subdirs = ['cache/'+d for d in os.listdir('cache/') if os.path.isdir('cache/'+d)]
    InverseDesignModels_folder = max(all_subdirs, key=os.path.getmtime)

   
# InverseDesign for scalar emissivity =========================================
if InverseDesign_scalar:
    #%% load ML models for the inverse design
    # this is the folder that contains all the variables
    InverseDesignModels_folder_here = InverseDesignModels_folder+'/scalar/'
    dirpath_old = os.getcwd()
    os.chdir(InverseDesignModels_folder_here)
    for filename in glob.glob("*.joblib"):
        variable_name = filename.replace(".joblib",''); # print(filename); 
        print(variable_name)
        globals()[variable_name] = joblib.load(filename)
    os.chdir(dirpath_old)    
    
    InverseDesigns_save_folder = InverseDesignModels_folder_here + '/InvDesRes/'
    os.makedirs(InverseDesigns_save_folder, exist_ok=True)
    InverseDesigns_save_filename = InverseDesigns_save_folder+date_for_dirname
    
    #%% run the inverse design
    for DT_or_DTGEN in DT_or_DTGEN__:
        if DT_or_DTGEN == 'DTGEN':            
            estimator_here = dt_gen            
            X_train_here = X_new_train
            y_train_here = y_new_train
        elif DT_or_DTGEN == 'DT':            
            estimator_here = dt            
            X_train_here = X_train
            y_train_here = y_train
        else:
            estimator_here = []
    
        feature_names = X_train_here.columns
        feature_set_mat = [x for x in feature_names if "Material" in x]
        feature_set_geom = [x for x in feature_names if "Geometry" in x]        
        
        # Do the inverse design for each geometry
        for target_scalar_emissivity in target_scalar_emissivity_:        
            for mat in feature_set_mat:        
                idx_mat = X_train_here[mat].astype(bool) 
                
                InverDesignSaveFilename_CSV = (InverseDesigns_save_filename + DT_or_DTGEN + '_'
                        + bestdesign_method + '_' + mat
                        + '_emiss_{0}'.format(target_scalar_emissivity))
                features_InverseDesign_here,min_max_dict,min_max_dict_orig = DoInverseDesign_SaveCSV(
                        estimator_here, X_train_here[idx_mat],
                        y_train_here[idx_mat], target_scalar_emissivity,
                        scaling_factors, gen_n=gen_n, plotting=True,
                        search_method = 'scalar_target', frequency=my_x,
                        n_best_candidates = 3,
                        bestdesign_method=bestdesign_method,
                        InverDesignSaveFilename_CSV=InverDesignSaveFilename_CSV,
                        n_best_leaves_force=n_best_leaves_force)
                features_InverseDesign_here = features_InverseDesign_here.astype(np.float)                                
              
                    
# InverseDesign for spectral emissivity =======================================
if InverseDesign_spectral:
    #%% load ML models for the inverse design
    # this is the folder that contains all the variables
    InverseDesignModels_folder_here = InverseDesignModels_folder+'/spectral/'
    dirpath_old = os.getcwd()
    os.chdir(InverseDesignModels_folder_here)
    for filename in glob.glob("*.joblib"):
        variable_name = filename.replace(".joblib",''); # print(filename); 
        print(variable_name)
        globals()[variable_name] = joblib.load(filename)
    os.chdir(dirpath_old)    
    
    InverseDesigns_save_folder = InverseDesignModels_folder_here + '/InvDesRes/'
    os.makedirs(InverseDesigns_save_folder, exist_ok=True)
    InverseDesigns_save_filename = InverseDesigns_save_folder+date_for_dirname
    
    #%% run the inverse design  
    if RadCooling:
        ## inverse design for radiative cooling
        # we need high emissivity between 8 and 13 microns
        
        title_here = 'RadCooling'        
        
        min_w_h = 1.45e14
        max_w_h = 2.35e14          
        
        vector_low_range = [[0,min_w_h], [max_w_h, 1e15]]
        low_emissivity_values = [0, 0]
        vector_high_range = [[min_w_h,max_w_h]]
        high_emissivity_values = [1]
        
        vector_low_high_range = [vector_low_range, vector_high_range]
        target_vector =  draw_target_spectrum_from_low_high_ranges(my_x, vector_low_high_range, [low_emissivity_values, high_emissivity_values], plotting = True)        
    
        for DT_or_DTGEN in DT_or_DTGEN__:
            if DT_or_DTGEN == 'DTGEN':                
                estimator_here = dt_gen                
                X_train_here = X_new_train
                y_train_here = y_new_train
            elif DT_or_DTGEN == 'DT':                
                estimator_here = dt
                X_train_here = X_train
                y_train_here = y_train
            else:
                estimator_here = []  
                
            feature_names = X_train_here.columns
            feature_set_mat = [x for x in feature_names if "Material" in x]
            feature_set_geom = [x for x in feature_names if "Geometry" in x]     
        
            for mat in feature_set_mat:        
                idx_mat = X_train_here[mat].astype(bool)
                
                #search_method = 'max_integrated_contrast' # just find the largest contrast
                #search_method = 'max_integrated_contrast_high_emiss' # just find the largest contrast
                InverDesignSaveFilename_CSV = InverseDesigns_save_filename+DT_or_DTGEN+'_'+bestdesign_method+'_'+mat+'_'+title_here
                features_InverseDesign_here,min_max_dict,min_max_dict_orig = DoInverseDesign_SaveCSV(estimator_here,X_train_here[idx_mat],y_train_here[idx_mat],vector_low_high_range, scaling_factors, gen_n=gen_n, plotting=True, search_method = 'max_integrated_contrast', frequency=my_x, n_best_candidates = 500, bestdesign_method=bestdesign_method, InverDesignSaveFilename_CSV = InverDesignSaveFilename_CSV, n_best_leaves_force=n_best_leaves_force)
                features_InverseDesign_here = features_InverseDesign_here.astype(np.float)
                feature_names_here = features_InverseDesign_here.columns
                
        
    if ChooseRandomFromTest_eachMat:
        ## inverse design for data in y_test #        
        feature_names = X_train_here.columns
        
        feature_set_mat = [x for x in feature_names if "Material" in x]
        
        # Do the inverse design for each geometry        
        for mat in feature_set_mat:        
            idx_mat_test = X_test[mat].astype(bool)                            
            rndm_trgt_idx =  np.random.choice(np.where(idx_mat_test)[0])
            vector = y_test[rndm_trgt_idx]
            vector = (vector.reshape(1,len(vector)))
            title_here = 'Test_'+str(rndm_trgt_idx)            
            print(mat)
            print(X_test.iloc[rndm_trgt_idx])
                
            for DT_or_DTGEN in DT_or_DTGEN__:
                if DT_or_DTGEN == 'DTGEN':
                    estimator_here = dt_gen                
                    X_train_here = X_new_train
                    y_train_here = y_new_train
                elif DT_or_DTGEN == 'DT':
                    estimator_here = dt                
                    X_train_here = X_train
                    y_train_here = y_train
                else:
                    estimator_here = []                
                
                idx_mat = X_train_here[mat].astype(bool)
                
                InverDesignSaveFilename_CSV = InverseDesigns_save_filename+DT_or_DTGEN+'_'+bestdesign_method+'_'+mat+'_'+title_here
                features_InverseDesign_here,min_max_dict,min_max_dict_orig = DoInverseDesign_SaveCSV(estimator_here,X_train_here[idx_mat],y_train_here[idx_mat],vector, scaling_factors, gen_n=gen_n, plotting=True, search_method = 'min_sum_square_error', frequency=my_x, n_best_candidates = 5, bestdesign_method=bestdesign_method, InverDesignSaveFilename_CSV = InverDesignSaveFilename_CSV)
                features_InverseDesign_here = features_InverseDesign_here.astype(np.float)
                feature_names_here = features_InverseDesign_here.columns
                
    if IndexTest_targets:
        for rndm_trgt_idx in idx_Test_targets:                    
            vector = y_test[rndm_trgt_idx]
            vector = (vector.reshape(1,len(vector)))
            title_here = 'Test_'+str(rndm_trgt_idx)            
            print(mat)
            print(X_test.iloc[rndm_trgt_idx])
                
            for DT_or_DTGEN in DT_or_DTGEN__:
                if DT_or_DTGEN == 'DTGEN':
                    estimator_here = dt_gen                
                    X_train_here = X_new_train
                    y_train_here = y_new_train
                elif DT_or_DTGEN == 'DT':
                    estimator_here = dt                
                    X_train_here = X_train
                    y_train_here = y_train
                else:
                    estimator_here = []                
                
                InverDesignSaveFilename_CSV = InverseDesigns_save_filename+DT_or_DTGEN+'_'+bestdesign_method+'_'+title_here
                features_InverseDesign_here,min_max_dict,min_max_dict_orig = DoInverseDesign_SaveCSV(estimator_here,X_train_here,y_train_here,vector, scaling_factors, gen_n=gen_n, plotting=True, search_method = 'min_sum_square_error', frequency=my_x, n_best_candidates = 5, bestdesign_method=bestdesign_method, InverDesignSaveFilename_CSV = InverDesignSaveFilename_CSV)
                features_InverseDesign_here = features_InverseDesign_here.astype(np.float)
                feature_names_here = features_InverseDesign_here.columns
        
              
    if random_multipeak:
        print('TO BE CODED, we wil do inverse design given two peaks. It is very similar code to RadCooling')
        
    if peak_at_lambda:
        
        
        title_here = 'peak_{0}um_wid_{1}rd'.format(target_peak_center_um, target_peak_width_rads)
        
        target_peak_center_rads = (([target_peak_center_um] * u.micron).to(
            u.Hz, equivalencies=u.spectral())*2*3.14).value
         
        
        min_w_h = target_peak_center_rads - target_peak_width_rads/2 ; min_w_h=min_w_h[0]
        max_w_h = target_peak_center_rads + target_peak_width_rads/2 ; max_w_h=max_w_h[0]
        
        vector_low_range = [[0,min_w_h], [max_w_h, 1e15]]
        low_emissivity_values = [0, 0]
        vector_high_range = [[min_w_h,max_w_h]]
        high_emissivity_values = [1]
        
        vector_low_high_range = [vector_low_range, vector_high_range]
        target_vector =  draw_target_spectrum_from_low_high_ranges(my_x,
                vector_low_high_range, [low_emissivity_values,
                    high_emissivity_values], plotting=True)        
    
        for DT_or_DTGEN in DT_or_DTGEN__:
            if DT_or_DTGEN == 'DTGEN':                
                estimator_here = dt_gen                
                X_train_here = X_new_train
                y_train_here = y_new_train
            elif DT_or_DTGEN == 'DT':                
                estimator_here = dt
                X_train_here = X_train
                y_train_here = y_train
            else:
                estimator_here = []  
                
            feature_names = X_train_here.columns
            feature_set_mat = [x for x in feature_names if "Material" in x]
            feature_set_geom = [x for x in feature_names if "Geometry" in x]     
        
            for mat in feature_set_mat:        
                idx_mat = X_train_here[mat].astype(bool)
                
                #search_method = 'max_integrated_contrast' # just find the largest contrast
                #search_method = 'max_integrated_contrast_high_emiss' # just find the largest contrast
                InverDesignSaveFilename_CSV = (InverseDesigns_save_filename + DT_or_DTGEN
                        + '_' + bestdesign_method + '_' + mat + '_' + title_here)
                features_InverseDesign_here,min_max_dict,min_max_dict_orig = DoInverseDesign_SaveCSV(
                        estimator_here, X_train_here[idx_mat],
                        y_train_here[idx_mat], vector_low_high_range,
                        scaling_factors, gen_n=gen_n, plotting=True,
                        search_method='max_integrated_contrast',
                        frequency=my_x, n_best_candidates=500,
                        bestdesign_method=bestdesign_method,
                        InverDesignSaveFilename_CSV=InverDesignSaveFilename_CSV,
                        n_best_leaves_force=n_best_leaves_force)
                features_InverseDesign_here = features_InverseDesign_here.astype(np.float)
                feature_names_here = features_InverseDesign_here.columns
