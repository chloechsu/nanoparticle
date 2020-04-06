from math import ceil
from time import time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def split_and_write_to_csv(data, filepath, rows_per_file=10000):
    if filepath.endswith('.csv'):
        filepath = filepath[:-4]
    number_of_chunks = int(ceil(float(data.shape[0]) / rows_per_file))
    for i, data_i in  enumerate(np.array_split(data, number_of_chunks)):
        p = '{}_{}-of-{}.csv'.format(filepath, i, number_of_chunks)
        if isinstance(data_i, pd.DataFrame):
            data_i.to_csv(p)
        else:
            np.savetxt(p, data_i, delimiter=',')
    print('Data saved to {}_?-of-{}.csv'.format(filepath, number_of_chunks))


#%% SAVING, while undoing scaling effects===============================================
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
