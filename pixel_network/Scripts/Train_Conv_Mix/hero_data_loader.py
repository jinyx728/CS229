######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
class HeroDataLoader():
    def __init__(self,dataloader,hero_dataloader):
        self.dataloader=dataloader
        self.hero_dataloader=hero_dataloader
        self.reset()
    
    def reset(self):
        self.iter=iter(self.dataloader)
        self.hero_iter=iter(self.hero_dataloader)


    def __iter__(self):
        return self

    def __next__(self):
        sample=self.iter.next()
        n_samples=len(sample['rotations'])
        while True:
            try:
                hero_sample=self.hero_iter.next()
                hero_offset_imgs=hero_sample['rotations']
                if len(hero_offset_imgs)<n_samples:
                    continue
                elif len(hero_offset_imgs)>n_samples:
                    for name in hero_sample:
                        hero_sample[name]=hero_sample[name][:n_samples]
                sample['hero']=hero_sample
                break
            except StopIteration:
                self.hero_iter=iter(self.hero_dataloader)

        return sample

    def __len__(self):
        return len(self.dataloader)