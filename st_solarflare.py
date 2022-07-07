import numpy as np
from pathlib import Path
import os
import random

from utils import MultivariateTransformer, save_transformer, save_shapelets_distances
from shapelets import ContractedShapeletTransform
from sktime_convert import from_3d_numpy_to_nested

#Load our training set
X_train= np.load(os.path.join('data', 'X_train_' + str(fold)+'.npy'))
y_train= np.load(os.path.join('data', 'y_train_' + str(fold)+'.npy'))
X_test= np.load(os.path.join('data', 'X_test_' + str(fold)+'.npy'))
y_test= np.load(os.path.join('data', 'y_test_' + str(fold)+'.npy'))

X_train = from_3d_numpy_to_nested(X_train)
y_train = np.asarray(y_train)
X_test = from_3d_numpy_to_nested(X_test)
y_test = np.asarray(y_test)

#Set lengths of shapelets to mine
min_length, max_length = 5, 30

print('here')

# How long (in minutes) to extract shapelets for.
# This is a simple lower-bound initially;
# once time is up, no further shapelets will be assessed
time_contract_in_mins = 220

time_contract_in_mins_per_dim = int(time_contract_in_mins/X_train.shape[1])

#If the time contract per dimensions is less than one minute, sample 
#time_contract_in_mins random dimensions and apply the ST to them
seed = 10

if time_contract_in_mins_per_dim < 1:
    random.seed(seed)
    dims = [random.randint(0, X_train.shape[1]-1) for p in range(0, int(time_contract_in_mins))]
        
    X_train = X_train.iloc[:, dims]
    
    #Spend one minute on each dimension
    time_contract_in_mins_per_dim = 1

# The initial number of shapelet candidates to assess per training series.
# If all series are visited and time remains on the contract then another
# pass of the data will occur
initial_num_shapelets_per_case = 10

# Whether or not to print on-going information about shapelet extraction.
# Useful for demo/debugging
verbose = 2
        
st = ContractedShapeletTransform(
    time_contract_in_mins=time_contract_in_mins_per_dim,
    num_candidates_to_sample_per_case=initial_num_shapelets_per_case,
    min_shapelet_length=min_length,
    max_shapelet_length=max_length,
    verbose=verbose,
    predefined_ig_rejection_level=0.001,
    max_shapelets_to_store_per_class=30
)

transformer = MultivariateTransformer(st)

transformer.fit(X_train, y_train)

X_new = transformer.transform(X_train)
    
name = 'solar_flare_s20220_maxsh30'

Path("results" + name).mkdir(parents=True, exist_ok=True)

save_shapelets_distances("results", name, transformer, test=False)
np.save("results" + name +"/" + name + "_X_new.npy", X_new)
save_transformer('results', name , transformer)

X_test_new = transformer.transform(X_test)

np.save("results" + name +"/" + name + "_X_test_new.npy", X_test_new)
save_shapelets_distances("results", name, transformer, test=True)