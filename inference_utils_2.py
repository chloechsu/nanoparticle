import os

import scipy
from scipy.io import loadmat
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from math import pi
import shelve
speed_of_light = 299792458.0

#sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#feature importance
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
#import pydotplus

def get_all_children(node_id,left,right):
    if left[node_id]==right[node_id]:
        return [node_id]
    else:
        left_children = get_all_children(left[node_id],left,right)
        right_children = get_all_children(right[node_id],left,right)
        return left_children+right_children






from time import strftime,time

from InverseDesign_utils_2 import gen_data_P1_P2_P3_Elzouka

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Latex
# setting plotting parameters
import matplotlib.pyplot as plt
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)             # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)        # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)      # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)      # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title
plt.rcParams["font.family"] = "Times"       # fontname

plt.rcParams["font.family"] = "Helvetica"



# should it me moved to the main inputs?


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

#%% SAVING ========================================================================================
def df_to_csv(df_in,filename, scaling_factors):
    """Undo the scaling effects and save the DataFrame to CSV"""
    df = df_in.copy()
        
    if 'Area/Vol' in df.columns:
        df['Area/Vol'] /= scaling_factors['Area/Volume']
    
    if 'log Area/Vol' in df.columns:
        AV_scaled = np.exp(df['log Area/Vol']); print(AV_scaled)
        AV = AV_scaled / scaling_factors['Area/Volume']; print(AV_scaled)
        df['log Area/Vol'] = np.log(AV)
        
        if not('Area/Vol' in df.columns):
            df['Area/Vol'] = np.exp(df['log Area/Vol'])
        
    if 'ShortestDim' in df.columns:
        df['ShortestDim'] /= scaling_factors['Length']
        
    if 'MiddleDim' in df.columns:
        df['MiddleDim'] /= scaling_factors['Length']
        
    if 'LongDim' in df.columns:
        df['LongDim'] /= scaling_factors['Length']
        
    df.to_csv(path_or_buf=filename);


#%% CURVE PARAMETRIZATION ==========================================================================
# function to convert from curve parmeters to actual curve
def onepeak_expNeg(height, width, xlocation, x):    
    B = np.log(1/0.01) / (width/2)**2    
    pk= height * np.exp( - B * (x-xlocation)**2)
    return pk

def multipeak_expNeg(param_peak, x):
    pk_all = np.zeros(np.size(x))
    kk = 0
    while kk < np.count_nonzero (param_peak):
        pk_all = pk_all + onepeak_expNeg(param_peak[kk], param_peak[kk+1], param_peak[kk+2], x)
        kk = kk + 3
    return pk_all
        
def z_poly_plus_peaks(x, parameters_all, n_poly):
    param_poly = parameters_all[0:n_poly]    
    param_peak = parameters_all[n_poly:]    
    y_fitted = np.polyval(param_poly, x) + multipeak_expNeg(param_peak, x)
    return y_fitted


def angular_freq_to_wavelength(freq_arr):
    return np.divide(2.0*pi*speed_of_light,freq_arr)





def get_Column_matStruct(MatStruct,ColumnName):
    """
    Inputs
    ------
    MatStruct: ndarray, each element represent data instance
    Example of how to get MatStruct
        MatStruct = loadmat('../data/All_data_08_13_19.mat')['All_data'][0]
        
    ColumnName: a string that exactly mathc the column name
    
    Returns
    -------
    list contains all the columns
    
    Usage
    -----
    # Asigning data to variables
    geom=get_Column_matStruct(MatStruct,'geom')
    material=get_Column_matStruct(MatStruct,'material')
    AV=get_Column_matStruct(MatStruct,'surface_over_volume')
    A=get_Column_matStruct(MatStruct,'A_surface')
    V=get_Column_matStruct(MatStruct,'Volume')
    ShortestDim=get_Column_matStruct(MatStruct,'ShortestDim')
    ShortestToLongestDim=get_Column_matStruct(MatStruct,'ShortestToLongestDim')
    MiddleDim=get_Column_matStruct(MatStruct,'MiddleDim')
    MiddleToLongestDim=get_Column_matStruct(MatStruct,'MiddleToLongestDim')
    LongestDim=get_Column_matStruct(MatStruct,'LongestDim')
    emiss_300K=get_Column_matStruct(MatStruct,'emissivity_total_W_300K')
    """
    varrr=[]
    allfields_name = MatStruct.dtype.names
    if ColumnName in str(allfields_name):
        ind_ColumnName = allfields_name.index(ColumnName)        
        for x in MatStruct[0]:    
            varrr.append(x[ind_ColumnName])
            
    return varrr


#utils
def get_material(x, scaling_factors):
    arr = get_Column_matStruct(x,'material')
    return [x[0] for x in arr]
def get_shape(x, scaling_factors):
    arr = get_Column_matStruct(x,'geom')
    return [x[0] for x in arr]
def get_emissivity_total_300K(x, scaling_factors):
    arr = get_Column_matStruct(x,'emissivity_total_W_300K')
    return [x[0][0] for x in arr]
def get_area(x, scaling_factors):
    arr = get_Column_matStruct(x,'A_surface')
    return [x[0][0]*scaling_factors['Area'] for x in arr]
def get_volume(x, scaling_factors):
    arr = get_Column_matStruct(x,'Volume')
    return [x[0][0]*scaling_factors['Volume'] for x in arr]
def get_volume_over_area(x, scaling_factors):
    arr = get_Column_matStruct(x,'Volume_over_surface')
    return [x[0][0]*scaling_factors['Volume/Area'] for x in arr]
def get_area_over_volume(x, scaling_factors):
    arr = get_Column_matStruct(x,'surface_over_volume')
    return [x[0][0]*scaling_factors['Area/Volume'] for x in arr]
def get_longest_dim(x, scaling_factors):
    arr = get_Column_matStruct(x,'LongestDim')
    return [x[0][0]*scaling_factors['LongestDim'] for x in arr]
def get_short_dim(x, scaling_factors):
    arr = get_Column_matStruct(x,'ShortestDim')
    return [x[0][0]*scaling_factors['ShortestDim'] for x in arr]
def get_middle_dim(x, scaling_factors):
    arr = get_Column_matStruct(x,'MiddleDim')
    return [x[0][0]*scaling_factors['MiddleDim'] for x in arr]
def get_short_to_long(x, scaling_factors):
    arr = get_Column_matStruct(x,'ShortestToLongestDim')
    return [x[0][0] for x in arr]
def get_mid_to_long(x, scaling_factors):
    arr = get_Column_matStruct(x,'MiddleToLongestDim')
    return [x[0][0] for x in arr]
def get_spectrum(x, scaling_factors):
    arr = get_Column_matStruct(x,'wrads_EmissHemisph')
    return arr
def get_spectrum_parameters(x, scaling_factors):
    arr = get_Column_matStruct(x,'poly_4_peaks_param_sorted')
    return arr
def get_peak_freq(x, scaling_factors): #from get_spectrum
    freq,height = x[:,0],x[:,1]
    return freq[np.argmax(height)] * scaling_factors['PeakFrequency']
def get_peak_emissivity(x, scaling_factors): #from get_spectrum
    return np.max(x[:,1]) * scaling_factors['PeakEmissivity']
def get_filepaths(x, scaling_factors):
    #determine if something is analytical
    resultfilepath = get_Column_matStruct(x,'ResultFilePath')
    mshfilepath = get_Column_matStruct(x,'mshfile_path')
    #return true if analytical
    return [False if (bool(x)&bool(y)) else True for x,y in zip(resultfilepath,mshfilepath)]
def get_runtime(data, scaling_factors):
    runtimes = get_Column_matStruct(data,'running_time_hr')
    return [x[0][0] if x else np.nan for x in runtimes]
#not sure if these are right anymore for 08/13/19 dataset
def get_lx(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'Lx')
    return [x[0][0]*scaling_factors['LengthX'] if x else np.nan for x in runtimes]
def get_height(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'h')
    return [x[0][0]*scaling_factors['Height'] if x else np.nan for x in runtimes]
def get_l(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'L')
    return [x[0][0]*scaling_factors['Length'] if x else np.nan for x in runtimes]

def get_Lx(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'Lx')
    return [x[0][0]*scaling_factors['Length'] if x else np.nan for x in runtimes]

def get_L(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'L')
    return [x[0][0]*scaling_factors['Length'] if x else np.nan for x in runtimes]

def get_w(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'w')
    return [x[0][0]*scaling_factors['Length'] if x else np.nan for x in runtimes]

def get_h(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'h')
    return [x[0][0]*scaling_factors['Length'] if x else np.nan for x in runtimes]

def get_D(x, scaling_factors):
    runtimes = get_Column_matStruct(x,'D')
    return [x[0][0]*scaling_factors['Length'] if x else np.nan for x in runtimes]

def is_spectrum_param(x, scaling_factors):
    # determine if the spectral curve here is parametrized
    get_Column_matStruct(x,'ResultFilePath')    
    #return true if the spectrum has parameters
    return [True if (bool(x)&bool(y)) else True for x,y in zip(resultfilepath,mshfilepath)]
    

#load data
def load_spectrum_param_data_mat(data, my_x, scaling_factors, return_dim_params=False):
    #get features
    area = get_area(data, scaling_factors)
    geom = get_shape(data, scaling_factors)
    vol = get_volume(data, scaling_factors)
    longdim = get_longest_dim(data, scaling_factors)
    shortdim = get_short_dim(data, scaling_factors)
    middledim = get_middle_dim(data, scaling_factors)
    material = get_material(data, scaling_factors)
    vol_area =  get_volume_over_area(data, scaling_factors)
    area_vol = get_area_over_volume(data, scaling_factors)
    emissivity = get_emissivity_total_300K(data, scaling_factors)
    short_to_long = get_short_to_long(data, scaling_factors)
    mid_to_long = get_mid_to_long(data, scaling_factors)
    Lx = get_Lx(data, scaling_factors)
    L = get_L(data, scaling_factors)
    w = get_L(data, scaling_factors)
    h = get_h(data, scaling_factors)
    D = get_D(data, scaling_factors)

    #get spectrum data
    spectrum = get_spectrum(data, scaling_factors)    
    spectrum_parameters = get_spectrum_parameters(data, scaling_factors)
    peak_frequency = [get_peak_freq(x, scaling_factors) for x in spectrum]
    peak_emissivity = [get_peak_emissivity(x, scaling_factors) for x in spectrum]
    #get runtime
    runtime = get_runtime(data, scaling_factors)
    #get analytial
    is_analytical = get_filepaths(data, scaling_factors)

    # assign array to fully define geometry dimension
    P1 = []; P2 = []; P3 = []    
    for gg,idx in zip(geom,range(len(geom))):
        if   gg == 'sphere':
            P1.append(D[idx])
            P2.append(0)
            P3.append(0)
        elif gg == 'wire':
            P1.append(D[idx])
            P2.append(Lx[idx])
            P3.append(0)
        elif gg == 'parallelepiped':
            P1.append(shortdim[idx])
            P2.append(middledim[idx])
            P3.append(longdim[idx])
        elif gg == 'TriangPrismIsosc':
            P1.append(h[idx])
            P2.append(L[idx])
            P3.append(Lx[idx])

    #get params
    if return_dim_params:
        lx_dim = get_lx(data, scaling_factors)
        height = get_height(data, scaling_factors)
        length_dim = get_l(data, scaling_factors)
        data_matrix = pd.DataFrame([is_analytical , middledim , longdim , lx_dim  , height , length_dim , geom     , shortdim    , short_to_long, mid_to_long, material , vol_area , area_vol , emissivity , peak_frequency , peak_emissivity , runtime ,  spectrum ,  spectrum_parameters ,  Lx ,  L ,  w ,  h ,  D ,  P1 ,  P2 ,  P3 ]).T
        data_matrix.columns =     ['is_analytical','MiddleDim','LongDim','LengthX','Height','Length'    ,'Geometry','ShortestDim','ShortToLong' ,'MidToLong' ,'Material','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime', 'spectrum', 'spectrum_parameters', 'Lx', 'L', 'w', 'h', 'D', 'P1', 'P2', 'P3']
        data_matrix[['MiddleDim','LongDim','LengthX','Height','Length','ShortestDim','ShortToLong','MidToLong','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime', 'spectrum', 'spectrum_parameters', 'Lx', 'L', 'w', 'h', 'D', 'P1', 'P2', 'P3']] = \
        data_matrix[['MiddleDim','LongDim','LengthX','Height','Length','ShortestDim','ShortToLong','MidToLong','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime', 'spectrum', 'spectrum_parameters', 'Lx', 'L', 'w', 'h', 'D', 'P1', 'P2', 'P3']].astype(np.float64)
    else:
        data_matrix = pd.DataFrame([ is_analytical , geom     , shortdim    ,  longdim , short_to_long  , mid_to_long , material , vol_area , area_vol , emissivity , peak_frequency  , peak_emissivity  , runtime  ,  spectrum ,  spectrum_parameters  , middledim ,  Lx ,  L ,  w ,  h ,  D ,  P1 ,  P2 ,  P3]).T
        data_matrix.columns =      ['is_analytical','Geometry','ShortestDim', 'LongDim', 'ShortToLong'  ,'MidToLong'  ,'Material','Vol/Area','Area/Vol','Emissivity','Peak_frequency' ,'Peak_emissivity' ,'Runtime' , 'spectrum', 'spectrum_parameters' ,'MiddleDim', 'Lx', 'L', 'w', 'h', 'D', 'P1', 'P2', 'P3']
        data_matrix[['ShortestDim','ShortToLong','MidToLong','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime', 'LongDim','MiddleDim', 'Lx', 'L', 'w', 'h', 'D', 'P1', 'P2', 'P3']] = \
        data_matrix[['ShortestDim','ShortToLong','MidToLong','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime', 'LongDim','MiddleDim', 'Lx', 'L', 'w', 'h', 'D', 'P1', 'P2', 'P3']].astype(np.float64)
    
    #one hot encode geometry,material
    data_matrix = pd.get_dummies(data_matrix,columns=['Geometry'])
    data_matrix = pd.get_dummies(data_matrix,columns=['Material'])
    
    print("Number of Samples: {0}".format(len(data_matrix)))
    assert len(data_matrix) == len(spectrum)
    
    #drop samples with spectra below 0
    drop_nonphysical_idx = []
    for spectra,idx in zip(spectrum,range(len(spectrum))):
        if len(np.where(spectra[:,1]<0)[0])!=0: #check if y value is below 0
            drop_nonphysical_idx.append(idx)
    data_matrix = data_matrix.drop(drop_nonphysical_idx)
    #drop samples in spectrum
    mask = np.ones(len(spectrum)).astype(bool)
    mask[drop_nonphysical_idx] = False
    spectrum = list(np.array(spectrum)[mask])
    
    ind_nonzero = np.array(np.nonzero(mask))
    if len(spectrum_parameters) > 0:
        spectrum_parameters = [spectrum_parameters[ii] for ii in ind_nonzero[0]]
    

    assert len(spectrum)==len(data_matrix)
    print("Removed {0} non-physical spectra".format(len(drop_nonphysical_idx)))
    print("New Shape: {0}".format(data_matrix.shape))
    
    #drop triangprism because deprecated geometry
    #check if deprecated data exists:
    if 'Geometry_TriangPrism' in data_matrix.columns:
        print("Removed {0} TriangPrism Geometries".format(data_matrix['Geometry_TriangPrism'].sum()))
        data_matrix = data_matrix[~data_matrix['Geometry_TriangPrism'].astype(bool)]
        data_matrix = data_matrix.drop("Geometry_TriangPrism",axis=1)
        print("New shape: {0}".format(data_matrix.shape))
    
    #check no duplicate x's in spectrum array
    non_dupl_spectrum = [a[np.unique(a[:, 0], return_index=True)[1]] for a in spectrum]
    assert not np.any([np.any(x[:,0][1:]<=x[:,0][:-1]) for x in non_dupl_spectrum]) #check no duplicate x values
    #get indices of spectrums with right length
    #limit to 10**15
    lower_bound = 8*10**14
    upper_bound = 200*10**14
    right_length_spectrum_idx = [True if (np.max(a[:,0])>lower_bound) and (np.max(a[:,0])<=upper_bound) else False for a in non_dupl_spectrum] # drop any samples with incomplete spectra
    non_dupl_right_length_spectrum = [a for a in non_dupl_spectrum if (np.max(a[:,0])>lower_bound)and(np.max(a[:,0])<=upper_bound)]
    data_matrix = data_matrix[right_length_spectrum_idx]
    print("Removed {0} Incomplete Spectra".format(len(right_length_spectrum_idx)-sum(right_length_spectrum_idx)))
    print("New Shape: {0}".format(data_matrix.shape))
    assert len(data_matrix) == len(non_dupl_right_length_spectrum)
    #interpolate_objects = [interp1d(np.array(x[:,0]),np.array(x[:,1]),fill_value='extrapolate',kind='slinear',bounds_error=False) for x in non_dupl_right_length_spectrum] #assume if outside range, then basically 0, easy to extrapolate    
    interpolate_objects = [interp1d(np.array(x[:,0]),np.array(x[:,1]),fill_value=0,kind='slinear',bounds_error=False) for x in non_dupl_right_length_spectrum] #assume if outside range, then basically 0, easy to extrapolate
    
    interpolated_ys = np.array([f(my_x) for f in interpolate_objects])
    
    #remove duplicates
    num_duplicates = sum(data_matrix.duplicated(['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
                                                 'Material_Au', 'Material_SiN', 'Material_SiO2',
                                                 'ShortestDim','ShortToLong','MidToLong','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime']))
    print("Removed {0} duplicates".format(num_duplicates))
    idx = ~data_matrix.duplicated(['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
                                                 'Material_Au', 'Material_SiN', 'Material_SiO2',
                                                 'ShortestDim','ShortToLong','MidToLong','Vol/Area','Area/Vol','Emissivity','Peak_frequency','Peak_emissivity','Runtime'])
    data_matrix = data_matrix[idx]
    print("New shape: {0}".format(data_matrix.shape))
    interpolated_ys = interpolated_ys[idx]
    #correct interpolation for non-physical spectra
    interpolated_ys[interpolated_ys<0] = 0.0

    idx_values=idx.values
    ind_nonzero = np.array(np.nonzero(idx_values))
    if len(spectrum_parameters) > 0:    
        spectrum_parameters = [spectrum_parameters[ii] for ii in ind_nonzero[0]]

    non_dupl_right_length_spectrum = [x for x,y in zip(non_dupl_right_length_spectrum,range(len(idx))) if idx.values[y]]
    assert len(data_matrix) == len(interpolated_ys)
    assert len(interpolated_ys) == len(non_dupl_right_length_spectrum)

    #could probably do this more efficiently with sklearn pipeline, but this will do for now
    #X_featurized contains X and feature engineering
    data_featurized = data_matrix.copy()
    #data_featurized['log Area'] = np.log(data_featurized['Area'])
    #data_featurized['log Volume'] = np.log(data_featurized['Volume'])
    data_featurized['log Area/Vol'] = np.log(data_featurized['Area/Vol'])
    data_featurized['log Vol/Area'] = np.log(data_featurized['Vol/Area'])

    data_featurized['log ShortToLong'] = np.log(data_featurized['ShortToLong'])
    data_featurized['log MidToLong'] = np.log(data_featurized['MidToLong'])
    data_featurized['log Emissivity'] = np.log(data_featurized['Emissivity'])
    data_featurized['log ShortestDim'] = np.log(data_featurized['ShortestDim'])
    data_featurized['log LongDim'] = np.log(data_featurized['LongDim'])
    data_featurized['log MiddleDim'] = np.log(data_featurized['MiddleDim'])
    
    return data_featurized,interpolated_ys,spectrum_parameters


def z_RF_DT_DTGEN_error_folds(X_reduced,y_reduced, feature_set, feature_set_dimensions, feature_set_geom_mat, data_featurized, my_x, \
                     num_folds=20, test_size=0.2, n_estimators=200, n_cpus = 1, keep_spheres = True, optional_title_folders='', \
                     use_log_emissivity=True, display_plots=True, display_txt_out = True, RF_or_DT__ = ['RF'], PlotTitle_extra = '', \
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
   
    All_errors = {}
    
    All_errors['RF'] = {}
    All_errors['DT'] = {}
    All_errors['DTGEN'] = {}
    
    All_errors['RF']['mae_matgeom'] = {}
    All_errors['RF']['r2_matgeom'] = {}
    All_errors['RF']['mse_matgeom'] = {}
    All_errors['RF']['rmse_matgeom'] = {}
    All_errors['RF']['Erel_matgeom'] = {}
    All_errors['RF']['ntest_matgeom'] = {}
    All_errors['RF']['ntrain_matgeom'] = {}    
    
    All_errors['DT']['mae_matgeom'] = {}
    All_errors['DT']['r2_matgeom'] = {}
    All_errors['DT']['mse_matgeom'] = {}
    All_errors['DT']['rmse_matgeom'] = {}
    All_errors['DT']['Erel_matgeom'] = {}
    All_errors['DT']['ntest_matgeom'] = {}
    All_errors['DT']['ntrain_matgeom'] = {}
    
    All_errors['DTGEN']['mae_matgeom'] = {}
    All_errors['DTGEN']['r2_matgeom'] = {}
    All_errors['DTGEN']['mse_matgeom'] = {}
    All_errors['DTGEN']['rmse_matgeom'] = {}
    All_errors['DTGEN']['Erel_matgeom'] = {}
    All_errors['DTGEN']['ntest_matgeom'] = {}
    All_errors['DTGEN']['ntrain_matgeom'] = {}
    
    
    
    
    pred_time = {}
    
    pred_time['RF'] = []
    pred_time['DT'] = []
    pred_time['DTGEN'] = []
    
    
    train_time = {}
    
    train_time['RF'] = []
    train_time['DT'] = []
    train_time['DTGEN'] = []
    
    
    
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
            
            # errors broken by material and geometry
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

    ## plotting the errors #####################################
    if display_plots:
        labelfontsize = 32
        axisfontsize = 26
        figsize = (16,10)
        
        PlotTitle_extra = RF_or_DT+' - '+PlotTitle_extra
        for error,error_name in zip([r2_matgeom,mae_matgeom,mse_matgeom,rmse_matgeom,Erel_matgeom],['R$^2$','MAE','MSE','RMSE','Erel']):
            width = 0.4
            x = [0,2,4]
            fig,ax = plt.subplots(figsize=figsize)
            for g,count in zip(geoms,range(len(geoms))):
                m_avg,m_std = [],[]
                for m in mats:
                    formal_material = m.split('_')[1]
                    formal_geom = g.split('_')[1]
                    key = formal_material + ' ' + formal_geom
                    this_error = error[key]
                    m_avg.append(np.average(this_error))
                    m_std.append(2*np.std(this_error))
                if display_plots:
                    ax.bar([a+width*count for a in x],m_avg,yerr=m_std, align='center', alpha=0.7, ecolor='black', capsize=6,width=width,label=formal_geom.capitalize())
            if display_plots:
                ax.legend(fontsize=labelfontsize)
                ax.set_ylabel(error_name,fontsize=labelfontsize)
                plt.title([feature_set_dimensions, PlotTitle_extra])
                plt.ylim(bottom=0)
                if error_name=='R$^2$':
                    plt.ylim(top=1)
                ax.set_xticks([float(a)+float(len(geoms)*width)/2.0 - width/2.0 for a in x])
                ax.set_xticklabels([m.split('_')[1] for m in mats])
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(axisfontsize) 
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(axisfontsize) 
        plt.show()
        
    if display_txt_out:
        print(feature_set_dimensions)
        display(Latex('R$^2$: {0:.6f} $\pm$ {1:.6f}'.format(np.average(r2),2*np.std(r2))))
        display(Latex('MAE: {0:.6f} $\pm$ {1:.6f}'.format(np.average(mae),2*np.std(mae))))
        display(Latex('RMSE: {0:.6f} $\pm$ {1:.6f}'.format(np.average(rmse),2*np.std(rmse))))
        display(Latex('MSE: {0:.6f} $\pm$ {1:.6f}'.format(np.average(mse),2*np.std(mse))))
        display(Latex('Erel: {0:.6f} $\pm$ {1:.6f}'.format(np.average(Erel),2*np.std(Erel))))

        print("average runtime per sample on lawrencium (clock seconds): {0}".format(data_featurized[data_featurized["Runtime"]>0]['Runtime'].mean()*3600))
        print("Average runtime per sample for "+RF_or_DT+" training feature(CPU seconds): {0}".format(np.average(train_time_feat)*n_cpus/10**6))
        print("Average runtime per sample for "+RF_or_DT+" inference feature prediction(CPU seconds): {0}".format(np.average(pred_time_feat)*n_cpus/10**6))
    
        
            