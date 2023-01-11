import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree
import joblib
from tqdm import tqdm


t = pd.read_csv("/Volumes/ExtremeSSD/UNI/thesis/ThesisData/data/Processed_data2.csv")


nDrops = len(t.loc[t.frame==0])
nFrames = max(t.frame)
print(f"nDrops:{nDrops}")
print(f"nFrames:{nFrames}")

r_c = np.array([[458, 498]])

@joblib.delayed
def computeRadialDistributionFunction(fromCentre, frame, COORDS, rList, dr, rho):
    coords = COORDS[frame*nDrops:(frame+1)*nDrops, :]

    # add coordinates of the centre to avoid division by zero errors
    # note it doesnt count since there's always a -1 in n1 and n2
    if fromCentre: coords = np.concatenate((coords, r_c), axis=0)
    
    kd = KDTree(coords)
    avg_n = np.zeros(len(rList))

    for i, r in enumerate(rList):
        
        if fromCentre:
            a = kd.query_ball_point(r_c, r + 20)
            b = kd.query_ball_point(r_c, r)
        else:
            a = kd.query_ball_point(coords, r + 20)
            b = kd.query_ball_point(coords, r)
        
        n1 = 0
        for j in a:
            n1 += len(j) - 1

        n2 = 0
        for j in b:
            n2 += len(j) - 1
        
        avg_n[i] = n1/len(a) - n2/len(b)

    if fromCentre: g2 = avg_n
    else: g2 = avg_n/(np.pi*(dr**2 + 2*rList*dr)*rho)
    return g2


fromCentre = True
dr = 5
rDisk = 822/2
if fromCentre: rList = np.arange(0, 2*rDisk, 1)
else: rList = np.arange(0, rDisk, 1)
rho = nDrops/(np.pi*rDisk**2)

COORDS = np.array(t.loc[:,["x","y"]])

parallel = joblib.Parallel(n_jobs = -2)
frames = 30000
g2 = parallel(
    computeRadialDistributionFunction(True, frame, COORDS, rList, dr, rho)
    for frame in tqdm( range(frames) )
)
g2 = np.array(g2)

saveData = False
if saveData:
    if fromCentre: pd.DataFrame(g2).to_csv(f'/Volumes/ExtremeSSD/UNI/thesis/ThesisData/data/g2_from_centre.csv', index=False)
    else: pd.DataFrame(g2).to_csv(f'/Volumes/ExtremeSSD/UNI/thesis/ThesisData/data/g2.csv', index=False)



