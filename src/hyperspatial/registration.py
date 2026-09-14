from __future__ import annotations
import numpy as np

def register_geometry(source: np.ndarray, target: np.ndarray, seed: int=42,
                      epochs: int=30, device: str="cpu", pre_registered: bool=False):
    """Geometry-only registration using the executed similarity/SVF implementation."""
    if source.shape[1]!=target.shape[1]: raise ValueError("source and target dimensions differ")
    if pre_registered:
        return source.astype(np.float32), {"mode":"pre_registered","seed":seed}
    import torch
    from run_mouse_temporal_transport_gate import best_rigid, chamfer
    from hyperspatial_molecular_hologram import RegistrationConfig, fit_geometry_field
    rigid,reflection,candidates=best_rigid(source,target)
    rng=np.random.default_rng(seed)
    source_fit=rigid[np.sort(rng.choice(len(rigid),min(5000,len(rigid)),replace=False))]
    target_fit=target[np.sort(rng.choice(len(target),min(5000,len(target)),replace=False))]
    field,history=fit_geometry_field(source_fit,target_fit,device,RegistrationConfig(
        epochs=epochs,batch_size=min(768,len(source_fit)),seed=seed))
    mapped=[]
    with torch.no_grad():
        for begin in range(0,len(rigid),4096):
            mapped.append(field(torch.as_tensor(rigid[begin:begin+4096],dtype=torch.float32,device=device)).cpu().numpy())
    mapped=np.vstack(mapped)
    return mapped.astype(np.float32),{"mode":"geometry","seed":seed,"reflection":reflection,
        "candidate_chamfers":candidates,"raw_chamfer":chamfer(source,target),
        "registered_chamfer":chamfer(mapped,target),"final_geometry_loss":float(history['loss'][-1]),
        "final_fold_loss":float(history['fold'][-1])}
