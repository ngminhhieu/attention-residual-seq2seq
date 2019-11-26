from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = read_csv('../data/data.csv')
    # shape (8759,1)
    data = data.groupby(['time'],as_index=False).agg({'load': 'sum'})
    load_data = data['load'].to_numpy()
    load_data = np.expand_dims(load_data, axis=1)
    np.savez('../data/data.npz', data = load_data)

