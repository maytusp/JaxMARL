import os 
import json 
import argparse
import numpy as np
from sklearn.cluster import KMeans
from stsc.stsc import self_tuning_spectral_clustering_np

# Interconvert between lists of clusters and arrays of cluster ids
def convert_clusters_to_labels(clusters):
    total = sum([len(x) for x in clusters])
    ret = np.zeros(total, dtype=int)
    for i,x in enumerate(clusters):
        ret[np.array(x)]=i 
    return ret.tolist()

def convert_labels_to_cluster(labels):
    return [labels[labels==l] for l in np.unique(labels)]

# Cross-play clustering
def similarity_from_data(data, epsilon=1e-10):
    # Cross play divided by self play
    sim = (data+data.T) / ((np.diag(data)[None]+np.diag(data)[:,None]))
    # Handle errors
    sim = np.where(np.isnan(sim),1,sim)
    # Clip to [0,1]
    sim = np.clip(sim,0,1)
    # Scale to [epsilon/2,1-epsilon/2] for stability
    sim = sim*(1-epsilon)+epsilon/2
    return sim 

def clusters_from_similarity(similarity_matrix, max_n_cluster = 5):
    clusters = self_tuning_spectral_clustering_np(similarity_matrix,max_n_cluster=max_n_cluster)
    return clusters 

# Self-play clustering
def get_pecan_clusters(data):
    "K-Means clustering on self-play returns"
    self_play = np.diag(data) if data.ndim==2 else data
    kmeans = KMeans(3)
    return kmeans.fit_predict(self_play[:,None]).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="large_room", help="Environment name")
    parser.add_argument("--infile", type=str, default="overcooked_cache/cross_play/cross_play_overcooked_large_room/avg.json", help="File to pull data from")  
    parser.add_argument("--outdir", type=str, default="overcooked_cache/clusters/large_room", help="Directory to output data to") 

    args = parser.parse_args()

    with open(args.infile,"r") as f:
        data = np.array(json.load(f))

    print("DATA")
    print(data)
    print()
    sim = similarity_from_data(data)
    print("SIM")
    print(sim.round(3))
    print()
    clusters = clusters_from_similarity(sim)
    clusters = convert_clusters_to_labels(clusters)
    print(f"TBS clusters {clusters}")

    pecan_clusters = get_pecan_clusters(data)
    print(f"PECAN clusters {pecan_clusters}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "clusters.json"), "w") as f:
        json.dump(clusters, f)
    with open(os.path.join(args.outdir, "pecan_clusters.json"), "w") as f:
        json.dump(pecan_clusters, f)