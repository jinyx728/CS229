######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
class MixDataLoader():
    def __init__(self,dataloaders):
        self.dataloaders=dataloaders
        self.reset()
    
    def reset(self):
        names=[]
        iters=[]
        loader_sizes=[]

        for name,loader in self.dataloaders.items():
            names.append(name)
            iters.append(iter(loader))
            loader_sizes.append(len(loader))

        self.names=names
        self.iters=iters
        self.loader_ids=self.rand_loader_ids(loader_sizes)
        self.num_batches=len(self.loader_ids)
        self.current=0

    def rand_loader_ids(self,loader_sizes):
        cumulative_n=[]
        total_n=0
        for n in loader_sizes:
            total_n+=n
            cumulative_n.append(total_n)

        def find_id(cumulative_n,n):
            for i in range(len(cumulative_n)):
                if n<cumulative_n[i]:
                    return i

        perm=np.random.permutation(total_n)
        loader_ids=[find_id(cumulative_n,p) for p in perm]

        return loader_ids

    def __iter__(self):
        return self

    def __next__(self):
        if self.current>=self.num_batches:
            raise StopIteration
        loader_id=self.loader_ids[self.current]
        loader_iter=self.iters[loader_id]
        sample=loader_iter.next()
        sample['label']=self.names[loader_id]
        self.current+=1
        return sample

    def __len__(self):
        return self.num_batches