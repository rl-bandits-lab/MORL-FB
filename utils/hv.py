import argparse
import numpy as np
from copy import deepcopy
from pygmo import hypervolume
import pygmo as pg
import copy


def compute_hypervolume_sparsity(obj_batch, ref_point):

    for i in range(len(ref_point)):
        obj_batch = obj_batch[obj_batch[:, i] < ref_point[i]]
        

    hv_cal = hypervolume(obj_batch)
    hv = hv_cal.compute(ref_point)

    contribute = hv_cal.contributions(ref_point)

    index = np.where(contribute > 0)

    obj_batch = obj_batch[index]

    sparsity = 0.0
    try:
        m = len(obj_batch[0])
        for dim in range(m):
            objs_i = np.sort(deepcopy(obj_batch.T[dim]))

            for i in range(1, len(objs_i)):
                sparsity += np.square(objs_i[i] - objs_i[i - 1])
        sparsity /= (len(obj_batch) - 1)
    except:
        pass

    return hv, sparsity


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--pref', type=str, required=True)
    parser.add_argument('--ref', type=float, nargs='+', default=[0, 0, -8000])

    args = parser.parse_args()

    data = np.load(args.data)
    pref_table = np.load(args.pref)

    uv_data = copy.deepcopy(data)


    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[i, j, k] = args.ref[k] * 2 - 1 * data[i, j, k]

    hvs, sparsities = [], []
    uts = []

    for i in range(data.shape[1]):
        hv, sparsity = compute_hypervolume_sparsity(data[:, i, :], args.ref)
        hvs.append(hv)
        sparsities.append(sparsity)
        ut = np.sum(uv_data[:, i, :]*pref_table, axis=1)
        uts.append(ut.mean())


    print(f'hv => {np.mean(hvs)}, {np.std(hvs)}')
    print(f'hvs => {hvs}')
    print(f'sparsity => {np.mean(sparsities)}, {np.std(sparsities)}')
    print(f'ut => {np.mean(uts)}, {np.std(uts)}')