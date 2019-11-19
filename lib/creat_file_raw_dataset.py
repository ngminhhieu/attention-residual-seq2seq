from pandas import read_csv
import numpy as np

if __name__ == "__main__":
    data = read_csv('../data/grid_data.csv', usecols=['NYISO'])
    # shape (8759,1)
    np.savez('../data/grid_data.npz', data = data)