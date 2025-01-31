import argparse
import numpy as np



def episodic_domination(base, others, pref):
    cnt = 0
    for (base_, others_, pref_) in zip(base, others, pref):
        dot_base = np.dot(base_, pref_)
        dot_others = np.dot(others_, pref_)
        if dot_others >= dot_base:
            cnt += 1

    return cnt / base.shape[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--others', type=str, required=True)
    parser.add_argument('--pref', type=str, required=True)

    args = parser.parse_args()


    base_data = np.load(args.base)
    others_data = np.load(args.others)
    pref = np.load(args.pref)

    eds = []

    for i in range(base_data.shape[1]):
        ed = episodic_domination(base_data[:, i, :], others_data[:, i, :], pref)
        eds.append(ed)

    print(f'Episodic domination: {np.mean(eds)} , {np.std(eds)}')