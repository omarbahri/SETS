import os
import numpy as np
import random
import pickle
from utils import get_all_shapelet_locations_scaled_threshold, get_all_shapelet_locations_scaled_threshold_test

random.seed(42)

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

name = 'solar_flare'
suffix = '_s20220_maxsh30'

X_train= np.load(os.path.join('data','X_train_strat.npy'))
y_train= np.load(os.path.join('data','y_train_strat.npy'))
X_test= np.load(os.path.join('data','X_test_strat.npy'))
y_test= np.load(os.path.join('data','y_test_strat.npy'))
    
st_dir = name + suffix

st_shapelets = load(os.path.join('results', st_dir, st_dir + '_shapelets.pkl'))
shapelets_distances = load(os.path.join('results', st_dir, st_dir + '_shapelets_distances.pkl'))
shapelets_distances_test = load(os.path.join('results', st_dir, st_dir + '_shapelets_distances_test.pkl'))

ts_length = X_train.shape[2]
occ_threshold = 1e-1

all_shapelet_locations, all_no_occurences, threshold = get_all_shapelet_locations_scaled_threshold(shapelets_distances, ts_length, occ_threshold/100.)
all_shapelet_locations_test, all_no_occurences_test = get_all_shapelet_locations_scaled_threshold_test(shapelets_distances_test, ts_length, threshold)

del shapelets_distances

all_shapelets_class_0 = []
all_shapelets_class_1 = []
all_shapelets_class_2 = []
all_shapelets_class_3 = []

all_heat_maps_0 = []
all_heat_maps_1 = []
all_heat_maps_2 = []
all_heat_maps_3 = []

for dim in range(X_test.shape[1]):
    print("Dim: ", dim)
    for index in sorted(all_no_occurences[dim], reverse=True):
        print(index)
        del st_shapelets[dim][index]
    
    #Get shapelets class occurences
    shapelets_classes = []
    
    for shapelet_locations in all_shapelet_locations[dim]:
        shapelet_classes = []
        for sl in shapelet_locations:
            shapelet_classes.append(y_train[sl[0]])
        shapelets_classes.append(shapelet_classes)
    
    shapelets_class_0 = []
    shapelets_class_1 = []  
    shapelets_class_2 = []
    shapelets_class_3 = []
    
    two_classes = []
    
    #Find shapelets that happen exclusively under one class
    for i, shapelet_classes in enumerate(shapelets_classes):
        # print(shapelet_classes)
        if not np.all(np.asarray(shapelet_classes)==0)\
        and not np.all(np.asarray(shapelet_classes)==1)\
        and not np.all(np.asarray(shapelet_classes)==2)\
        and not np.all(np.asarray(shapelet_classes)==3):
            two_classes.append(i)
    
    for index in sorted(two_classes, reverse=True):
        print(index)
        del st_shapelets[dim][index]
        del all_shapelet_locations[dim][index]
        try:
            del all_shapelet_locations_test[dim][index]
        except Exception:
            pass
        del shapelets_classes[index]
        
        
    for i, shapelet_classes in enumerate(shapelets_classes):
        if np.all(np.asarray(shapelet_classes)==0):
            shapelets_class_0.append(i)
        elif np.all(np.asarray(shapelet_classes)==1):
            shapelets_class_1.append(i)
        elif np.all(np.asarray(shapelet_classes)==2):
            shapelets_class_2.append(i)
        elif np.all(np.asarray(shapelet_classes)==3):
            shapelets_class_3.append(i)
        else:
            print('None: something\'s wrong')
    
    all_shapelets_class_0.append(shapelets_class_0)
    all_shapelets_class_1.append(shapelets_class_1)
    all_shapelets_class_2.append(shapelets_class_2)
    all_shapelets_class_3.append(shapelets_class_3)
    
    #Get shapelet_locations distributions per exclusive class
    heat_maps_0 = {}
    
    for s in shapelets_class_0:
        heat_map = np.zeros(ts_length)
        num_occurences=0
        
        for sl in all_shapelet_locations[dim][s]: 
            for idx in range(sl[1],sl[2]):
                heat_map[idx] += 1
            num_occurences += 1
        
        heat_map = heat_map/num_occurences
        heat_maps_0[s] = heat_map
    
    heat_maps_1 = {}
        
    for s in shapelets_class_1:
        heat_map = np.zeros(ts_length)
        num_occurences=0
        
        for sl in all_shapelet_locations[dim][s]: 
            for idx in range(sl[1],sl[2]):
                heat_map[idx] += 1
            num_occurences += 1
        
        heat_map = heat_map/num_occurences
        heat_maps_1[s] = heat_map
    
    heat_maps_2 = {}
        
    for s in shapelets_class_2:
        heat_map = np.zeros(ts_length)
        num_occurences=0
        
        for sl in all_shapelet_locations[dim][s]: 
            for idx in range(sl[1],sl[2]):
                heat_map[idx] += 1
            num_occurences += 1
        
        heat_map = heat_map/num_occurences
        heat_maps_2[s] = heat_map
        
    heat_maps_3 = {}
        
    for s in shapelets_class_3:
        heat_map = np.zeros(ts_length)
        num_occurences=0
        
        for sl in all_shapelet_locations[dim][s]: 
            for idx in range(sl[1],sl[2]):
                heat_map[idx] += 1
            num_occurences += 1
        
        heat_map = heat_map/num_occurences
        heat_maps_3[s] = heat_map
        
    all_heat_maps_0.append(heat_maps_0)
    all_heat_maps_1.append(heat_maps_1)
    all_heat_maps_2.append(heat_maps_2)
    all_heat_maps_3.append(heat_maps_3)
    
for i, hm_i in enumerate([all_heat_maps_0, all_heat_maps_1, all_heat_maps_2, all_heat_maps_3]):
    np.save(os.path.join('results', st_dir, 'util_data', 'all_heat_maps_' +\
                         str(i) + '.npy'), hm_i)
    
for i, sc_i in enumerate([all_shapelets_class_0, all_shapelets_class_1, all_shapelets_class_2, all_shapelets_class_3]):
    np.save(os.path.join('results', st_dir, 'util_data', 'all_shapelets_class_' +\
                         str(i) + '.npy'), sc_i)
        
np.save(os.path.join('results', st_dir, 'util_data',\
                     'all_shapelet_locations.npy'), all_shapelet_locations)
    
np.save(os.path.join('results', st_dir, 'util_data',\
                     'all_shapelet_locations_trth_test.npy'), all_shapelet_locations_test)
    
np.save(os.path.join('results', st_dir, 'util_data',\
                     'all_no_occurences_trth_test.npy'), all_no_occurences_test)