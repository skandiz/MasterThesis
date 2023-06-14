
import numpy as np
import pandas as pd
from tqdm import tqdm
import graph_tool.all as gt

def motif_search(mofitList, sizeList, g):
    counts = np.zeros(len(mofitList), dtype=int)
    for i, mofit in enumerate(mofitList):
        _, temp = gt.motifs(g, sizeList[i], motif_list=[mofit])
        counts[i] = temp[0]
    return counts

rawTrajs = pd.read_parquet("../tracking/stardist_res/sharp/df_linked.parquet")
res_path = "stardist_results" 
analysis_data_path = "stardist_analysis_data"

nFrames = 30000
frames = np.arange(0, nFrames, 1)
mean_d = 2*rawTrajs.groupby("frame").mean().r.values

sizeList = [3, 4, 5, 5]
motifList = []
for i in range(len(sizeList)):
    motifList.append(gt.load_graph(f'{res_path}/graph/motif/selected/motif_{i}.graphml'))

for factor in np.linspace(1.2, 3, 10):
    motif_results = np.zeros((nFrames, len(motifList)), dtype=int)
    for frame in tqdm(frames):
        X = np.array(rawTrajs.loc[rawTrajs.frame == frame, ['x', 'y']])
        g, pos = gt.geometric_graph(X, mean_d[frame]*factor)
        motif_results[frame] = motif_search(motifList, sizeList, g)
    df_motif = pd.DataFrame(motif_results, columns=[f'motif_{i}' for i in range(len(motifList))])
    df_motif.to_parquet(f"{analysis_data_path}/motif_analysis_factor{factor}.parquet")


