# SETS
This is the accompanying repository of our paper ["Shapelet-based Temporal Association Rule Mining for Multivariate Time Series Classification"](https://kdd-milets.github.io/milets2022/papers/MILETS_2022_paper_7874.pdf) presented at the SIGKDD 2022 Workshop on Mining and Learning from Time Series (MiLeTS).

### Installation: <br />
The packages required to run this code are listed in `requirements.txt`. 
To create a new virtual environment and install them:
```
python3 -m pip install --user virtualenv
python3 -m venv sets
source sets/bin/activate
pip install -r requirements.txt
```
### Instructions: <br />
The solar flare dataset is provided in the `data/sf` directory.<br />
`sets.sh` runs SETS on the solar flare dataset as described in the paper. Feel free to experiment with different datasets and parameters.<br />
To use a custom dataset, split it into train and test sets as 3D Numpy arrays with shape `(N,D,L)`, such that `N` is the number of time series instances, `D` is the number of dimensions, and `L` is the time series length, and save it in a new directory under `data`.
```
chmod +x sets.sh
./sets.sh
```
For large datasets, and depending on the time contract, parts of SETS might take longer to run. `sets.sh` keeps intermediary results to allow reusing them if needed.
