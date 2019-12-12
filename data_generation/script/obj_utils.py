import numpy as np
def get_edge_from_fcs(fcs):
    n_vts=np.max(fcs)+1
    def hash(i0,i1):
        if i0<i1:
            return i0*n_vts+i1
        else:
            return i1*n_vts+i0
    def dehash(h):
        return h//n_vts,h%n_vts
    edge_hash_set=set()
    for i0,i1,i2 in fcs:
        edge_hash_set.add(hash(i0,i1))
        edge_hash_set.add(hash(i1,i2))
        edge_hash_set.add(hash(i2,i0))

    return [dehash(h) for h in edge_hash_set]