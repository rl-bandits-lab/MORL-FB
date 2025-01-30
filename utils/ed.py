import argparse
import numpy as np



def episodic_domination(fb, pd, pref):
    cnt = 0
    for (fb_, pd_, pref_) in zip(fb, pd, pref):
        dot_fb = np.dot(fb_, pref_)
        dot_pd = np.dot(pd_, pref_)
        if dot_pd >= dot_fb:
            cnt += 1

    return cnt / fb.shape[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--others', type=str, required=True)
    parser.add_argument('--pref', type=str, required=True)

    args = parser.parse_args()


    fb_data = np.load(args.fb)
    pd_data = np.load(args.pd)
    pref = np.load(args.pref)

    eds = []

    for i in range(fb_data.shape[1]):
        ed = episodic_domination(fb_data[:, i, :], pd_data[:, i, :], pref)
        eds.append(ed)

    print(f'Episodic domination: {np.mean(eds)} , {np.std(eds)}')