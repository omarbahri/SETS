import os
import numpy as np
import random
import itertools
from tslearn.neighbors import KNeighborsTimeSeries

random.seed(42)

name = 'solar_flare'
suffix = '_s20220_maxsh30'

X_train= np.load(os.path.join('data','X_train_strat.npy'))
y_train= np.load(os.path.join('data','y_train_strat.npy'))
X_test= np.load(os.path.join('data','X_test_strat.npy'))
y_test= np.load(os.path.join('data','y_test_strat.npy'))
        
st_dir = name + suffix

st_shapelets = np.load(os.path.join('results', st_dir, st_dir + '_shapelets.pkl'),\
                       allow_pickle=True)
all_shapelet_locations = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_shapelet_locations.npy'), allow_pickle=True)
all_shapelet_locations_test = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_shapelet_locations_trth_test.npy'), allow_pickle=True)
all_shapelets_class_0 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_shapelets_class_0.npy'), allow_pickle=True)
all_shapelets_class_1 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_shapelets_class_1.npy'), allow_pickle=True)
all_shapelets_class_2 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_shapelets_class_2.npy'), allow_pickle=True)
all_shapelets_class_3 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_shapelets_class_3.npy'), allow_pickle=True)
all_heat_maps_0 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_heat_maps_0.npy'), allow_pickle=True)
all_heat_maps_1 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_heat_maps_1.npy'), allow_pickle=True)
all_heat_maps_2 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_heat_maps_2.npy'), allow_pickle=True)
all_heat_maps_3 = np.load(os.path.join('results', st_dir, 'util_data',\
                    'all_heat_maps_3.npy'), allow_pickle=True)     

series_len = X_train.shape[2]

#Sort dimensions by their highest shapelet scores 
all_shapelets_scores = np.load(os.path.join('results', st_dir,\
                    'solar_flare_s20220_pyts_scores.npy'), allow_pickle=True)

for dim in range(len(st_shapelets)):
    shapelets_best_scores.append(max(all_shapelets_scores[dim]))

shapelets_best_scores = np.argsort(shapelets_best_scores)[::-1]

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(random_state=1, max_iter=100, hidden_layer_sizes=240, learning_rate_init=1e-3).fit(X_train.reshape((X_train.shape[0],-1)), y_train)

y_pred = model.predict(X_test.reshape((X_test.shape[0],-1)))

label_0 = 0
label_1 = 1
label_2 = 2
label_3 = 3

###HERE
if label_0 == 1:
    y_pred = y_pred+1
    
all_shapelets_ranges_0 = []
all_shapelets_ranges_1 = []
all_shapelets_ranges_2 = []
all_shapelets_ranges_3 = []

for dim in range(X_test.shape[1]):    
    shapelets_ranges_0 = {}
    shapelets_ranges_1 = {}
    shapelets_ranges_2 = {}
    shapelets_ranges_3 = {}
    
    #Get [min,max] of each shapelet occurences
    for label, (all_shapelets_class, shapelets_ranges) in enumerate(zip([all_shapelets_class_0,all_shapelets_class_1,\
                                            all_shapelets_class_2,all_shapelets_class_3],\
                                            [shapelets_ranges_0,shapelets_ranges_1,\
                                            shapelets_ranges_2,shapelets_ranges_3])):
                
        for j, sls in enumerate([all_shapelet_locations[dim][i]\
                                 for i in all_shapelets_class[dim]]):
            s_mins = 0
            s_maxs = 0
            n_s = 0
            
            for sl in sls:
                ts = X_train[sl[0]][dim][sl[1]:sl[2]]
                
                s_mins += ts.min()
                s_maxs += ts.max()
                n_s += 1
                
            shapelets_ranges[all_shapelets_class[dim][j]] = (s_mins/n_s, s_maxs/n_s)
    
    all_shapelets_ranges_0.append(shapelets_ranges_0)
    all_shapelets_ranges_1.append(shapelets_ranges_1)
    all_shapelets_ranges_2.append(shapelets_ranges_2)
    all_shapelets_ranges_3.append(shapelets_ranges_3)

ts_length = X_test.shape[2]
    
def get_shapelets_locations_test(idx, all_sls, dim, all_shapelets_class):
    all_locs = {}
    try: 
        for i, s in enumerate([all_sls[dim][j]\
                                 for j in all_shapelets_class[dim]]):
            i_locs = []
            for loc in s:
                # print(loc)
                if loc[0] == idx:
                    loc = (loc[1],loc[2])
                    i_locs.append(loc)
            all_locs[i] = i_locs
    except Exception:
        print('')
    return all_locs

##Optimize by fitting outside or returning a list of all nns at once
def get_nearest_neighbor(knn, X_test, idx):
    # pred_label = y_pred[idx]
    pred_label = y_test[idx]
    target_labels = np.argwhere(y_train!=pred_label)
    
    X_test_knn = X_test[idx].reshape(1, X_test.shape[1], X_test.shape[2])
    X_test_knn = np.swapaxes(X_test_knn, 1, 2)
    
    _, nn = knn.kneighbors(X_test_knn)
    nn_idx = target_labels[nn][0][0][0]
    
    return nn_idx

knn_0 = KNeighborsTimeSeries(n_neighbors=1)
X_train_knn = X_train[np.argwhere(y_train==label_0)].reshape(np.argwhere(y_train==label_0).shape[0],\
                                X_train.shape[1], X_train.shape[2])
X_train_knn = np.swapaxes(X_train_knn, 1, 2)
knn_0.fit(X_train_knn)

knn_1 = KNeighborsTimeSeries(n_neighbors=1)
X_train_knn = X_train[np.argwhere(y_train==label_1)].reshape(np.argwhere(y_train==label_1).shape[0],\
                                X_train.shape[1], X_train.shape[2])
X_train_knn = np.swapaxes(X_train_knn, 1, 2)
knn_1.fit(X_train_knn)

knn_2 = KNeighborsTimeSeries(n_neighbors=1)
X_train_knn = X_train[np.argwhere(y_train==label_2)].reshape(np.argwhere(y_train==label_2).shape[0],\
                                X_train.shape[1], X_train.shape[2])
X_train_knn = np.swapaxes(X_train_knn, 1, 2)
knn_2.fit(X_train_knn)

knn_3 = KNeighborsTimeSeries(n_neighbors=1)
X_train_knn = X_train[np.argwhere(y_train==label_3)].reshape(np.argwhere(y_train==label_3).shape[0],\
                                X_train.shape[1], X_train.shape[2])
X_train_knn = np.swapaxes(X_train_knn, 1, 2)
knn_3.fit(X_train_knn)
    
for instance_idx in range(0,X_test.shape[0]):
    orig_label = y_pred[instance_idx]
    
    for target_label in set(np.unique(y_train)) - set([orig_label]):
    
        print("instance_idx: " + str(instance_idx))
        print("from: " + str(orig_label) + " to: " + str(target_label))
        
        if orig_label == label_0:
            original_all_shapelets_class = all_shapelets_class_0
            original_shapelets_ranges = all_shapelets_ranges_0
        if orig_label == label_1:
            original_all_shapelets_class = all_shapelets_class_1
            original_shapelets_ranges = all_shapelets_ranges_1
        if orig_label == label_2:
            original_all_shapelets_class = all_shapelets_class_2
            original_shapelets_ranges = all_shapelets_ranges_2
        if orig_label == label_3:
            original_all_shapelets_class = all_shapelets_class_3
            original_shapelets_ranges = all_shapelets_ranges_3
        
        if target_label == label_0:
            all_target_shapelets_class = all_shapelets_class_0
            all_target_heat_maps = all_heat_maps_0
            target_knn = knn_0
            target_shapelets_ranges = all_shapelets_ranges_0
        if target_label == label_1:
            all_target_shapelets_class = all_shapelets_class_1
            all_target_heat_maps = all_heat_maps_1
            target_shapelets_ranges = all_shapelets_ranges_1
            target_knn = knn_1
        if target_label == label_2:
            all_target_shapelets_class = all_shapelets_class_2
            all_target_heat_maps = all_heat_maps_2
            target_shapelets_ranges = all_shapelets_ranges_2
            target_knn = knn_2
        if target_label == label_3:
            all_target_shapelets_class = all_shapelets_class_3
            all_target_heat_maps = all_heat_maps_3
            target_shapelets_ranges = all_shapelets_ranges_3
            target_knn = knn_3
            
        nn_idx = get_nearest_neighbor(target_knn, X_test, instance_idx)
        
        print('nn_idx ', nn_idx)
        
        naive_cf_dims = np.zeros((len(shapelets_best_scores), series_len))
               
        for dim in shapelets_best_scores:
            naive_cf = X_test[instance_idx].copy()
            naive_cf_pred = model.predict(naive_cf.reshape(1,-1))
                                                  
            if label_0 == 1:
                naive_cf_pred += 1
            
            if y_pred[instance_idx] == naive_cf_pred:
                print('Moving to dimension: ', dim)
                #Get the locations where the original class shapelets occur    
                all_locs = get_shapelets_locations_test(instance_idx, all_shapelet_locations_test,\
                                                   dim, original_all_shapelets_class)
                    
                #Replace the original class shapelets with nn values
                for c_i in all_locs:
                    for loc in all_locs.get(c_i):
                        naive_cf_pred = model.predict(naive_cf.reshape(1,-1))
                                                      
                        if label_0 == 1:
                            naive_cf_pred += 1
                        
                        if y_pred[instance_idx] == naive_cf_pred:
                            print('Removing original shapelet')
                            nn = X_test[nn_idx].reshape(-1)
                            
                            target_shapelet = nn[loc[0]:loc[1]]
                                                    
                            s_min = target_shapelet.min()
                            s_max = target_shapelet.max()
                            t_min = naive_cf[dim][loc[0]:loc[1]].min()
                            t_max = naive_cf[dim][loc[0]:loc[1]].max()
                            
                            if s_max-s_min==0:
                                target_shapelet = (t_max+t_min)/2*np.ones(len(target_shapelet))
                            else:
                                target_shapelet = (t_max-t_min)*(target_shapelet-s_min)/(s_max-s_min)+t_min
                            
                            start = loc[0]
                            end = loc[1]
                            
                            naive_cf[dim][start:end] = target_shapelet                     
                
                #Introduce new shapelets from the target class
                for _, target_shapelet_idx in enumerate(all_target_heat_maps[dim]):            
                    naive_cf_pred = model.predict(naive_cf.reshape(1,-1))
                    if label_0 == 1:
                        naive_cf_pred += 1
                    
                    if y_pred[instance_idx] == naive_cf_pred:
                        print('Introducing new shapelet')
                        
                        h_m = all_target_heat_maps[dim].get(target_shapelet_idx)   
                        
                        center = (np.argwhere(h_m>0)[-1][0] - np.argwhere(h_m>0)[0][0])//2 + np.argwhere(h_m>0)[0][0]
                        
                        target_shapelet = st_shapelets[dim][target_shapelet_idx][0]
                        print('dim, target sh idx ', dim, target_shapelet_idx)
                        target_shapelet_length = target_shapelet.shape[0]
                        
                        start = center - target_shapelet_length//2
                        end = center + (target_shapelet_length-target_shapelet_length//2)
                        
                        if start < 0:
                            end = end - start
                            start = 0
                        
                        if end > ts_length:
                            start = start - (end - ts_length + 1)
                            end = ts_length - 1
                                                
                        s_min = target_shapelet.min()
                        s_max = target_shapelet.max()
                        t_min = naive_cf[dim][start:end].min()
                        t_max = naive_cf[dim][start:end].max() 
                                                
                        if s_max-s_min==0:
                            target_shapelet = (t_max+t_min)/2*np.ones(len(target_shapelet))
                        else:
                            target_shapelet = (t_max-t_min)*(target_shapelet-s_min)/(s_max-s_min)+t_min
                                                
                        naive_cf[dim][start:end] = target_shapelet
                                                            
            #Save the perturbed dimension
            naive_cf_dims[dim] = naive_cf[dim]
            
            naive_cf_pred = model.predict(naive_cf.reshape(1,-1))
                        
            if label_0 == 1:
                naive_cf_pred += 1
            
            if y_pred[instance_idx] != naive_cf_pred:    
                print('cf found')
                                        
                if not os.path.exists(os.path.join('results', st_dir, 'sets', 'sorted_dims_combs_pred')):
                    os.makedirs(os.path.join('results', st_dir, 'sets', 'sorted_dims_combs_pred'))
                    
                np.save(os.path.join('results', st_dir, 'sets', 'sorted_dims_combs_pred', str(instance_idx) +\
                        '_to_' + str(target_label) + '.npy'), naive_cf)
                    
                break
            
            else:
                print("Trying dims combinations")
                #Try all combinations of dimensions
                for L in range(0, len(shapelets_best_scores)+1):
                    if y_pred[instance_idx] == naive_cf_pred:
                        for subset in itertools.combinations(shapelets_best_scores, L):
                            if y_pred[instance_idx] == naive_cf_pred:
                                if len(subset)>=2:
                                    naive_cf = X_test[instance_idx].copy()
                                    for dim_ in subset:
                                        naive_cf[dim_] = naive_cf_dims[dim_]
                                    
                                    naive_cf_pred = model.predict(naive_cf.reshape(1,-1))
                                    if label_0 == 1:
                                        naive_cf_pred += 1
                
                                    if y_pred[instance_idx] != naive_cf_pred:
                                        print('cf found')
                                        print('final dims: ', subset)
                                        break
                            else:
                                break
                    else:
                        break
                    

            if y_pred[instance_idx] != naive_cf_pred:    
                if not os.path.exists(os.path.join('results', st_dir, 'sets', 'sorted_dims_combs_pred')):
                    os.makedirs(os.path.join('results', st_dir, 'sets', 'sorted_dims_combs_pred'))
                    
                np.save(os.path.join('results', st_dir, 'sets', 'sorted_dims_combs_pred', str(instance_idx) +\
                        '_to_' + str(target_label) + '.npy'), naive_cf)
                break
        
    
    