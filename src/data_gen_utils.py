# -*- coding: utf-8 -*-

# DT_compression_Elzouka.py
from time import time
import numpy as np
import pandas as pd
from math import pi

### --- Generating Random Samples --- ###
def gen_data_P1_P2_P3_Elzouka(X,n=10**5):
    """This is different from "gen_data". Here, random pick is based on the geometry parameters P1,P2 and P3.
    After picking all the data, we will calculate the required feature columns    
    """
    start_time = time()

    # initializing the X_gen
    cols = list(X.columns)
    X_gen = pd.DataFrame(columns=cols)
    X_gen[cols[0]] = np.zeros(n)    

    # picking material, uniform random
    feature_set_mat = [x for x in cols if "Material" in x]    
    mat_idx = np.random.choice(np.arange(len(feature_set_mat)),size=n)        
    mat = pd.get_dummies(pd.Series(mat_idx)).values
    X_gen[feature_set_mat] = mat    
    
    # picking geometry, uniform random
    feature_set_geom = [x for x in cols if "Geometry" in x]    
    geom_idx = np.random.choice(np.arange(len(feature_set_geom)),size=n)
    geom = pd.get_dummies(pd.Series(geom_idx)).values
    
    X_gen[feature_set_geom] = geom
    
    # for every geometry, pick P1, P2 and P3
    features_to_be_set = ['P1','P2','P3']        
    for geom_here,ii_geom in zip(feature_set_geom,np.arange(len(feature_set_geom))):        
        idx_geom_here = np.argwhere(geom_idx==ii_geom)[:,0]        
        if len(idx_geom_here) > 0:
            X_geom_here = X[X[geom_here].astype(bool)]        

            # setting P1, P2 and P3
            for feature_here in features_to_be_set: 
                X_gen.loc[idx_geom_here, feature_here] = random_draw_my_number_list(
                        X_geom_here, feature_here, len(idx_geom_here))        
            X_gen = X_gen.astype('float64')                

            # from P1, P2 and P3, calculate A/V, Shortest, Middle and Longest
            if "sphere" in geom_here:                
                P1 = np.array(X_gen.loc[idx_geom_here, 'P1']); D = P1; r = D/2                        
                X_gen.loc[idx_geom_here, "ShortestDim"] = P1            
                X_gen.loc[idx_geom_here, "MiddleDim"] = P1
                X_gen.loc[idx_geom_here, "LongDim"] = P1            

                Area = 4*pi* r**2
                Volume = 4/3*pi* r**3

            elif "wire" in geom_here:                
                P1 = np.array(X_gen.loc[idx_geom_here, 'P1']); D = P1; r = D/2            
                P2 = np.array(X_gen.loc[idx_geom_here, 'P2']); Lx = P2;            

                param_wire = np.column_stack((Lx,D,D))                        
                param_wire_sorted = param_wire
                param_wire_sorted.sort(axis=1)                                    
                X_gen.loc[idx_geom_here, "ShortestDim"] = param_wire_sorted[:,0]
                X_gen.loc[idx_geom_here, "MiddleDim"]   = param_wire_sorted[:,1]
                X_gen.loc[idx_geom_here, "LongDim"]     = param_wire_sorted[:,2]            

                Area   = 2*(0.25*pi*D**2) + Lx*pi*D
                Volume = 0.25*pi* D**2 * Lx

            elif "parallelepiped" in geom_here:                
                P1 = np.array(X_gen.loc[idx_geom_here, 'P1']); w = P1
                P2 = np.array(X_gen.loc[idx_geom_here, 'P2']); h = P2
                P3 = np.array(X_gen.loc[idx_geom_here, 'P3']); Lx = P3
                
                param_wire = np.column_stack((P1,P2,P3))                        
                param_wire_sorted = param_wire
                param_wire_sorted.sort(axis=1) 

                X_gen.loc[idx_geom_here, "ShortestDim"] = param_wire_sorted[:,0]
                X_gen.loc[idx_geom_here, "MiddleDim"]   = param_wire_sorted[:,1]
                X_gen.loc[idx_geom_here, "LongDim"]     = param_wire_sorted[:,2]

                Area=2*(w+h)*Lx + 2*w*h
                Volume=w*h*Lx

            elif "TriangPrismIsosc" in geom_here:
                P1 = np.array(X_gen.loc[idx_geom_here, 'P1']); h = P1
                P2 = np.array(X_gen.loc[idx_geom_here, 'P2']); L = P2
                P3 = np.array(X_gen.loc[idx_geom_here, 'P3']); Lx = P3

                param_wire = np.column_stack((h, L, Lx))                        
                param_wire_sorted = param_wire
                param_wire_sorted.sort(axis=1)

                X_gen.loc[idx_geom_here, "ShortestDim"] = param_wire_sorted[:,0]
                X_gen.loc[idx_geom_here, "MiddleDim"]   = param_wire_sorted[:,1]
                X_gen.loc[idx_geom_here, "LongDim"]     = param_wire_sorted[:,2]

                Area=2*(0.5*L*h) + Lx*L + 2*Lx*np.sqrt((L/2)**2+h**2)
                Volume=0.5*L*h*Lx        

            A_V = Area/Volume          
            X_gen.loc[idx_geom_here, 'Area'] = Area
            X_gen.loc[idx_geom_here, 'Volume'] = Volume
            X_gen.loc[idx_geom_here, 'Area/Vol'] = A_V
            X_gen.loc[idx_geom_here, 'log Area/Vol'] = np.log(A_V)

    end_time = time()
    print("Total Time to generate {0:,} samples: {1:.4f} seconds".format(n,end_time-start_time))
    return X_gen


def random_draw_my_number_list(X, feature_name, n):
    avg_diff_in_spacing = np.diff(np.sort(X[feature_name])).mean()
    feature_values = np.random.choice(X[feature_name],size=n) + np.random.normal(scale=avg_diff_in_spacing/2,size=n)
    feature_values[feature_values<X[feature_name].min()] = X[feature_name].min()
    feature_values[feature_values>X[feature_name].max()] = X[feature_name].max()    
    return feature_values

