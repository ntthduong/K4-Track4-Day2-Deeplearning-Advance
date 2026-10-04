"""Forward-only latency with warmup and GPU synchronization."""
import copy
import time
import numpy as np
import torch

def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError('Use >=10 warmup and >=50 measured iterations')
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    return dict(p50=float(p50), p95=float(p95), p99=float(p99), mean=float(np.mean(samples)), n=iters)

def latency_report(model, batch_size, img_size, dtype='fp32', device='cuda', warmup=10, iters=100, k_views=1):
    if dtype not in ('fp32', 'amp', 'fp16') or k_views < 1:
        raise ValueError('Invalid dtype or view count')
    device = torch.device(device)
    clone = copy.deepcopy(model).to(device).eval()
    if dtype == 'fp16':
        clone.half()
    else:
        clone.float()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == 'fp16' else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device.type, enabled=dtype == 'amp' and device.type == 'cuda'):
            for i in range(k_views):
                clone(x.flip(-1) if i % 2 else x)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else None
    result = bench(forward, warmup, iters, sync)
    return {**result, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'dtype': dtype, 'batch': batch_size, 'img_size': img_size, 'K': k_views,
            'images_per_s': batch_size * 1000 / result['p50'], 'torch': str(torch.__version__),
            'preprocessing': False, 'scope': 'forward; aggregation excluded', 'warmup': warmup}

def tta_latency(model, k_views, **kw):
    return latency_report(model, k_views=k_views, **kw)
