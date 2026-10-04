"""timm models, frozen backbones and optimizer groups."""
import copy
import torch
from torch import nn

SUGGESTED_BACKBONES = {'resnet50': 'resnet50', 'resnext50': 'resnext50_32x4d', 'convnext_tiny': 'convnext_tiny',
                       'deit_small': 'deit_small_patch16_224', 'swin_tiny': 'swin_tiny_patch4_window7_224',
                       'efficientnet_b0': 'efficientnet_b0', 'mobilenetv3': 'mobilenetv3_large_100'}

def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init='finetune'):
    import timm
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(f'Unknown initialization: {init}')
    model = timm.create_model(SUGGESTED_BACKBONES.get(name, name), pretrained=pretrained and init != 'scratch',
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == 'frozen':
        freeze_backbone(model)
    return model

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    model.eval()
    model.get_classifier().train()

def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = {id(p) for p in model.get_classifier().parameters()}
    # Separate head norm/bias too: no weight decay on any norm/bias.
    groups = {(is_head, decay): [] for is_head in (False, True) for decay in (False, True)}
    no_decay = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[(id(p) in head, p.ndim > 1 and not name.endswith('.bias') and name not in no_decay)].append(p)
    return [{'params': ps, 'lr': lr_head if h else lr_backbone, 'weight_decay': weight_decay if d else 0.}
            for (h, d), ps in groups.items() if ps]

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_gmacs(model, img_size=224):
    # thop may omit some attention ops: label the counting tool in the report.
    from thop import profile
    clone = copy.deepcopy(model).cpu().eval()
    with torch.inference_mode():
        macs, _ = profile(clone, inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)
    return macs / 1e9
