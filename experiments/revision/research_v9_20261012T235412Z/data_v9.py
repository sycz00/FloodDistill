"""Only replace targets BEFORE shared geometric augmentation."""
from pathlib import Path
import common_v9 as c

def dataset(spec,arm,seed):
    import torch
    from data.pseudo_labeled import PseudoLabeledDataset
    primary=arm in ['human','teacher']
    base=dict(pseudo_dir=spec['pseudo'],n_samples=2500,use_curated=True,index_file=spec['index'],seed=seed,extra_keys=frozenset(),mult_noise=True,geom_aug=True)
    if primary:base.update(pseudo_dir=str(c.ROOT/'source_images'),n_samples=252,index_file=str(c.ROOT/'source_images/index.json'))
    class Targets(PseudoLabeledDataset):
        def __getitem__(self,i):
            a=torch.load(self.files[i],weights_only=True);mask=a['mask'].clone()
            if primary and arm=='teacher':
                b=torch.load(c.ROOT/'source_teacher'/self.files[i].name,weights_only=True);mask=b['mask'].long().clone()
                assert torch.equal(mask!=-1,a['mask']!=-1),'paired valid support changed'
            elif not primary and arm!='ce':
                b=torch.load(c.ROOT/'edge_labels'/self.files[i].name,weights_only=True);mask=b[arm].long().clone()
            x=self._apply_mult_noise(a['image'],i)
            return self._apply_geom(dict(image=x,mask=mask),i)
    return Targets(**base)
