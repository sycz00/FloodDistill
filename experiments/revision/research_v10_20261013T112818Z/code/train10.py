"""CPU adapters around the frozen Trainer; no independent training stack."""
import argparse,time,math
from pathlib import Path
import common10 as c

def loaders(spec,method,seed,FP_parent=None):
    torch,cj,old,v=c.setup()
    from torch.utils.data import DataLoader
    if method=='SUP':
        from data.sen1floods11 import get_dataloaders
        dl,_,_=get_dataloaders(spec['roots']['s11'],batch_size=16,num_workers=6,img_size=224,eval_img_size=512,drop_last=True)
        assert len(dl.dataset)==252
        dl.dataset.transform.set_random_seed(seed)
        return DataLoader(dl.dataset,batch_size=16,shuffle=True,num_workers=6,pin_memory=False,drop_last=True,persistent_workers=True)
    from data.pseudo_labeled import PseudoLabeledDataset
    if method=='FP_KD':
        with torch.random.fork_rng(devices=[]):fp_model=c.load(FP_parent)
    else:fp_model=None
    class Cached(PseudoLabeledDataset):
        def __getitem__(self,i):
            if method=='FP_KD':
                item=torch.load(self.files[i],weights_only=True);x=self._apply_mult_noise(item['image'],i);result=self._apply_geom(dict(image=x,mask=item['mask'],clean=item['image'].clone()),i)
                with torch.no_grad():result['soft_logits']=fp_model(result.pop('clean')[None])[0].float()
                return result
            if method.startswith('F'):
                item=torch.load(self.files[i],weights_only=True);target=torch.load(c.DATA/'mixed_targets'/self.files[i].name,weights_only=True)[method];x=self._apply_mult_noise(item['image'],i);return self._apply_geom(dict(image=x,mask=item['mask'],soft_logits=torch.stack([1-target,target])),i)
            if method!='KDoff':return super().__getitem__(i)
            assert c.read(c.ROOT/'teacher_cache_policy.json',{}).get('mode')=='float32_eight_geometric_views'
            item=torch.load(self.files[i],weights_only=True);x=self._apply_mult_noise(item['image'],i)
            marker=torch.tensor([[0,1],[2,3]])
            result=self._apply_geom(dict(image=x,mask=item['mask'],view=marker),i)
            view=result.pop('view');code=None
            for flip in [0,1]:
                for rotation in range(4):
                    candidate=(marker.flip(-1) if flip else marker).rot90(rotation,(-2,-1))
                    if torch.equal(view,candidate):code=4*flip+rotation
            assert code is not None
            cached=torch.load(c.DATA/'teacher_logits'/self.files[i].name,weights_only=True)
            result['soft_logits']=cached['logits'][code].clone().float()
            return result
    ds=Cached(spec['pseudo'],n_samples=2500,seed=seed,use_curated=True,index_file=spec['index'],extra_keys=frozenset(),mult_noise=True,geom_aug=True)
    return DataLoader(ds,batch_size=16,shuffle=True,num_workers=2,pin_memory=False,drop_last=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    manifest=c.read(c.ROOT/'manifest.json');job=manifest['training'][args.job]
    torch,cj,old,v=c.setup();assert not torch.cuda.is_available(),'V10 training is CPU-only'
    from config import TrainConfig
    from utils.seeding import seed_basic
    from utils.metrics import accumulate_metrics,finalize_metrics
    from torch import nn
    seed=job['seed'];graph=job['graph'];method=job['method'];is_qat=method in ['d0','d1','d2'];
    if graph=='SELECTED':graph=c.read(c.ROOT/'smallest_viable_selection.json')['graph']
    spec=c.read(c.V9/'training_spec.json');spec.update(seed=seed,workers=2)
    run=c.ROOT/('smokes' if args.smoke else 'runs')/args.job
    assert not (run.exists() and any(run.iterdir())),'Use a distinct attempt for nonempty run directories'
    run.mkdir(parents=True,exist_ok=True);tic=time.monotonic();seed_basic(seed)
    if graph!='UNet':
        teacher=old.teacher_cpu(spec);del teacher
    model=c.build(graph,pretrained=graph=='UNet');qat_parent=None;FP_parent=None
    if is_qat:
        qp=c.read(c.ROOT/'qat_parents'/f"{graph}_{job['precision']}_s{seed}.json");assert qp,'Prepared QAT parent required';qat_parent=qp['entry'];FP_parent=qp['FP_parent'];model=c.load(qat_parent).requires_grad_(True)
        from models.compression_quant import compression_train_mode
        compression_train_mode(model,enabled=method!='d0')
    if job.get('parent'):
        parent=c.read(c.ROOT/'scientific_receipts'/f"{graph}_KDoff_s{seed}.json")
        model.load_state_dict(torch.load(parent['best_path'],map_location='cpu',weights_only=True)['model'],strict=True)
    initial=c.state_hash(model.state_dict())
    # Model graph/pretraining must not change the shared cohort's minibatch stream.
    seed_basic(seed);val,test=cj.loaders(spec)
    if job.get('cohort'):
        choice=c.read(c.ROOT/'negative_cohorts.json')[job['cohort']];spec['index']=choice['index'];spec['pseudo']=choice['pseudo']
    train=loaders(spec,'KDoff' if method in ['KDoff','d2'] else 'FP_KD' if method=='d1' else 'CE' if is_qat else method,seed,FP_parent=FP_parent);per_epoch=len(train)
    if is_qat:epochs,frequency,patience,smoothing,clip,lossname=10,2,0,1,1.,'focal'
    elif graph=='UNet':epochs,frequency,patience,smoothing,clip,lossname=50,2,20,3,0.,'ce'
    elif method=='SUP':epochs,frequency,patience,smoothing,clip,lossname=math.ceil(7800/per_epoch),max(1,round(312/per_epoch)),10**6,1,5.,'focal'
    else:epochs,frequency,patience,smoothing,clip,lossname=50,2,0,1,5.,'focal'
    limit=50 if args.smoke else None
    class Observed:
        def __init__(self,dl):self.dl=dl;self.first_images=[];self.first_batches=[];self.seen=0
        def __getattr__(self,k):return getattr(self.dl,k)
        def __len__(self):return min(len(self.dl),max(0,limit-self.seen)) if limit else len(self.dl)
        def __iter__(self):
            iterator=iter(self.dl)
            for i in range(len(self)):
                batch=next(iterator);self.seen+=1
                if len(self.first_images)<2:
                    self.first_images.append(c.state_hash({'image':batch['image']}));self.first_batches.append(c.state_hash({k:batch[k] for k in ['image','mask']}))
                yield batch
    train=Observed(train)
    class Offline(nn.Module):
        gamma=2.;kd_valid_only=True
        def forward(self,z,x,y,precomputed_teacher_logits=None):
            assert precomputed_teacher_logits is not None
            value=v.offline_kd_loss(z,y,precomputed_teacher_logits)
            return value,dict(offline_loss=float(value.detach()))
    cfg=TrainConfig(epochs=math.ceil(50/per_epoch) if args.smoke else epochs,lr=job['lr'],weight_decay=1e-5 if is_qat else .05,task_loss=lossname,use_amp=False,grad_clip=clip,early_stop_patience=patience,val_smoothing_window=smoothing,val_every_n_epochs=1 if args.smoke else frequency,defer_terminal_evaluation=True,out_dir=str(run),seed=seed,gpu_ids=[0],deterministic=graph=='UNet')
    metadata=dict(quantized=is_qat,precision=job.get('precision'),cohort=job.get('cohort'),fp_reference=FP_parent['id'] if FP_parent else None,control='QAT_'+method if is_qat else 'cohort_'+job['cohort'] if job.get('cohort') else None,graph=graph,method=method,seed=seed,lr=job['lr'],initial_model_sha256=initial,training_images=len(train.dataset),updates_per_epoch=per_epoch,updates_required=1560 if is_qat else 750 if graph=='UNet' else 7800,recipe='TableI retained launcher defaults' if graph=='UNet' else 'v5 SUP focal7800' if method=='SUP' else 'v9 R4 focal7800; KDoff reference formula',source_only_selection=True,cache_policy=c.read(c.ROOT/'teacher_cache_policy.json',{}),device='cpu',smoke=args.smoke)
    class SoftTargets(nn.Module):
        gamma=2.;kd_valid_only=True
        def forward(self,z,x,y,precomputed_teacher_logits=None):
            assert precomputed_teacher_logits is not None
            value=v.soft_focal_loss(z,precomputed_teacher_logits[:,1],y!=-1);return value,dict(soft_focal=float(value.detach()))
    class ParentDistill(nn.Module):
        gamma=2.;kd_valid_only=True
        def __init__(self):
            super().__init__();self.parent=c.load(FP_parent).eval().requires_grad_(False)
        def forward(self,z,x,y,precomputed_teacher_logits=None):
            with torch.no_grad():targets=self.parent(x)
            value=v.offline_kd_loss(z,y,targets);return value,dict(parent_KD=float(value.detach()))
    class Trainer(cj.CompressionTrainer):
        def _init_wandb(self):return None
        def _train_epoch_supervised(self,model,optimizer):
            if is_qat:compression_train_mode(model,enabled=method!='d0')
            return super()._train_epoch_supervised(model,optimizer)
        def _train_epoch_distill(self,model,optimizer):
            if is_qat:compression_train_mode(model,enabled=method!='d0')
            return super(cj.CompressionTrainer,self)._train_epoch_distill(model,optimizer)
        def save_checkpoint(self,model,optimizer,epoch,metrics,filename='checkpoint.pt',**kw):
            path=super().save_checkpoint(model,optimizer,epoch,metrics,filename,**kw)
            payload=torch.load(path,map_location='cpu',weights_only=False);payload['v10_metadata']=metadata
            import os
            temporary=str(path)+'.metadata.tmp';torch.save(payload,temporary);os.replace(temporary,path);return path
        def _build_optimizer(self,model,config):
            opt=super()._build_optimizer(model,config)
            def guard(optimizer,*_):
                seen=False
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is not None:
                            seen=True;assert torch.isfinite(parameter.grad).all(),'Nonfinite gradient'
                assert seen,'No student gradient'
            opt.register_step_pre_hook(guard);return opt
        @torch.no_grad()
        def evaluate(self,model,loader):
            model.eval();cells=None
            for j,b in enumerate(loader):
                if args.smoke and j:break
                z=model(b['image'].float());cj.finite_output(model,(b['image'],),z);cells=accumulate_metrics(z,b['mask'].long(),2,-1,cells)
            result=finalize_metrics(cells);result['water_IoU']=result['iou_per_class'][1];return result
    trainer=Trainer(model,train,val,test,cfg,loss_fn=Offline() if method in ['KDoff','d1','d2'] else SoftTargets() if method.startswith('F') else None)
    assert trainer.device.type=='cpu'
    result=trainer.fit();best=run/'best.pt';last=run/'last.pt';bp=torch.load(best,map_location='cpu',weights_only=False);lp=torch.load(last,map_location='cpu',weights_only=False)
    receipt=dict(outcome='completed',status='complete',**metadata,best_path=str(best),best_sha256=c.sha(best),last_path=str(last),last_sha256=c.sha(last),completed_updates=lp['training_state']['global_step'],selected_updates=bp['training_state']['global_step'],completed_epochs=lp['epoch']+1,selected_epoch=bp['epoch']+1,source_validation=result,epochs=trainer.epoch_records,first_image_hashes=train.first_images,first_minibatch_hashes=train.first_batches,elapsed_seconds=time.monotonic()-tic,parameters=sum(p.numel() for p in model.parameters()))
    if not args.smoke and graph!='UNet':assert receipt['completed_updates']==(1560 if is_qat else 7800)
    if args.smoke:assert receipt['completed_updates']==50
    c.save(c.ROOT/('smoke_receipts' if args.smoke else 'scientific_receipts')/(args.job+'.json'),receipt)
if __name__=='__main__':main()
