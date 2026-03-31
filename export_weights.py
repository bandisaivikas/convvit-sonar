"""
export_weights.py  —  Fixed version
=====================================
Place this file in the same folder as best_sonar_convvit.pt
Then run:  python export_weights.py

All variable names printed as they are written.
Matches exactly what convvit_top.cpp expects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path('best_sonar_convvit.pt')
CACHE_PATH = Path('cache/spectrograms.npz')
OUT_DIR    = Path('fpga_convvit/weights')
TB_DIR     = Path('fpga_convvit/tb')

# ── MODEL DEFINITION (self-contained, no import) ──────────────────────────────
NUM_CLASSES = 5
NUM_PATCHES = 152
IMG_H, IMG_W = 128, 304
pH, pW = 8, 19

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
    def forward(self, x): return self.pool(self.conv(x))

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(1,   32, pool=True)
        self.stage2 = ConvBlock(32,  64, pool=True)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((pH, pW)))
    def forward(self, x):
        return self.stage3(self.stage2(self.stage1(x)))

class DynamicScaleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, dynamic=True):
        super().__init__()
        self.H, self.Dh = num_heads, embed_dim // num_heads
        self.dynamic = dynamic
        self.qkv   = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.proj  = nn.Linear(embed_dim, embed_dim)
        self.drop  = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1))
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.H,self.Dh).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2,-1)
        if self.dynamic:
            drange = attn.abs().amax(dim=(-2,-1),keepdim=True).clamp(min=1.0)
            attn = (attn / drange) * self.alpha
        else:
            attn = attn * (self.Dh ** -0.5)
        w   = self.drop(F.softmax(attn, dim=-1))
        out = (w @ v).transpose(1,2).reshape(B,N,D)
        return self.proj(out), w

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, dynamic=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = DynamicScaleAttention(embed_dim, num_heads, dropout, dynamic)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_d = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_d, embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        a, w = self.attn(self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, w

class ConvViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=256, depth=3,
                 num_heads=8, mlp_ratio=4.0, dropout=0.1, dynamic=True):
        super().__init__()
        self.cnn       = CNNBackbone()
        self.proj      = nn.Linear(128, embed_dim)
        N              = NUM_PATCHES
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, N+1, embed_dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dynamic)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim//2, num_classes))
    def forward(self, x, return_attn=False):
        B  = x.size(0)
        f  = self.cnn(x); f = f.flatten(2).transpose(1, 2); f = self.proj(f)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, f], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        aws = []
        for blk in self.blocks:
            x, w = blk(x); aws.append(w)
        out = self.head(self.norm(x)[:, 0])
        return (out, aws) if return_attn else out

# ── LOAD ──────────────────────────────────────────────────────────────────────
print('='*60)
print('  ConvViT Weight Export')
print('='*60)

if not MODEL_PATH.exists():
    print(f'ERROR: {MODEL_PATH} not found')
    print(f'Run this from the folder containing best_sonar_convvit.pt')
    sys.exit(1)

model = ConvViT(dynamic=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()   # IMPORTANT: sets BN to use running stats

nm = dict(model.named_modules())   # name -> module lookup

total = sum(p.numel() for p in model.parameters())
print(f'Model loaded: {total:,} params\n')

# ── HELPERS ───────────────────────────────────────────────────────────────────
def quantize_int8(tensor):
    """Quantize float tensor to INT8. Returns flat int8 numpy array."""
    t = tensor.detach().float().numpy().flatten()
    scale = max(float(np.abs(t).max()) / 127.0, 1e-8)
    return np.clip(np.round(t / scale), -128, 127).astype(np.int8)

def get_conv_w(path):
    return quantize_int8(nm[path].weight)

def get_bn(path):
    """Returns (running_mean, running_var, gamma, beta) as float32 arrays."""
    bn = nm[path]
    return (bn.running_mean.detach().numpy().astype(np.float32),
            bn.running_var.detach().numpy().astype(np.float32),
            bn.weight.detach().numpy().astype(np.float32),
            bn.bias.detach().numpy().astype(np.float32))

def get_linear_w(path):
    return quantize_int8(nm[path].weight)

def get_linear_b(path):
    m = nm[path]
    if m.bias is None:
        return np.zeros(m.out_features, dtype=np.float32)
    return m.bias.detach().numpy().astype(np.float32)

# ── WRITE HELPERS ─────────────────────────────────────────────────────────────
def write_int8(f, name, arr):
    """Write INT8 C array. Prints name on success."""
    flat = np.array(arr).flatten().astype(np.int8)
    f.write(f'static const int8_t {name}[{len(flat)}] = {{\n')
    for i in range(0, len(flat), 16):
        chunk = flat[i:i+16]
        f.write('    ' + ', '.join(f'{int(v):4d}' for v in chunk))
        if i + 16 < len(flat):
            f.write(',')
        f.write('\n')
    f.write('};\n\n')
    f.flush()
    print(f'  wrote {name} [{len(flat)}]')

def write_float(f, name, arr):
    """Write float C array. Prints name on success."""
    flat = np.array(arr).flatten().astype(np.float32)
    f.write(f'static const float {name}[{len(flat)}] = {{\n')
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        f.write('    ' + ', '.join(f'{float(v):.8f}f' for v in chunk))
        if i + 8 < len(flat):
            f.write(',')
        f.write('\n')
    f.write('};\n\n')
    f.flush()
    print(f'  wrote {name} [{len(flat)}]')

def write_scalar(f, name, val):
    f.write(f'static const float {name} = {float(val):.8f}f;\n\n')
    f.flush()
    print(f'  wrote {name} = {float(val):.6f}')

# ── MAIN EXPORT ───────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

h_path = OUT_DIR / 'weights.h'
print(f'Writing {h_path} ...\n')

with open(h_path, 'w') as f:

    f.write('// AUTO-GENERATED by export_weights.py\n')
    f.write('// ConvViT weights — matches convvit_top.cpp naming exactly\n')
    f.write('#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n#include <stdint.h>\n\n')

    # ── CNN Stage 1 ───────────────────────────────────────────────────────────
    print('[CNN Stage 1]')
    f.write('// CNN Stage 1\n')
    write_int8 (f, 'W_S1_C1',       get_conv_w('cnn.stage1.conv.0'))
    bn = get_bn('cnn.stage1.conv.1')
    write_float(f, 'W_S1_BN1_MEAN', bn[0])
    write_float(f, 'W_S1_BN1_VAR',  bn[1])
    write_float(f, 'W_S1_BN1_G',    bn[2])
    write_float(f, 'W_S1_BN1_B',    bn[3])
    write_int8 (f, 'W_S1_C2',       get_conv_w('cnn.stage1.conv.3'))
    bn = get_bn('cnn.stage1.conv.4')
    write_float(f, 'W_S1_BN2_MEAN', bn[0])
    write_float(f, 'W_S1_BN2_VAR',  bn[1])
    write_float(f, 'W_S1_BN2_G',    bn[2])
    write_float(f, 'W_S1_BN2_B',    bn[3])

    # ── CNN Stage 2 ───────────────────────────────────────────────────────────
    print('\n[CNN Stage 2]')
    f.write('// CNN Stage 2\n')
    write_int8 (f, 'W_S2_C1',       get_conv_w('cnn.stage2.conv.0'))
    bn = get_bn('cnn.stage2.conv.1')
    write_float(f, 'W_S2_BN1_MEAN', bn[0])
    write_float(f, 'W_S2_BN1_VAR',  bn[1])
    write_float(f, 'W_S2_BN1_G',    bn[2])
    write_float(f, 'W_S2_BN1_B',    bn[3])
    write_int8 (f, 'W_S2_C2',       get_conv_w('cnn.stage2.conv.3'))
    bn = get_bn('cnn.stage2.conv.4')
    write_float(f, 'W_S2_BN2_MEAN', bn[0])
    write_float(f, 'W_S2_BN2_VAR',  bn[1])
    write_float(f, 'W_S2_BN2_G',    bn[2])
    write_float(f, 'W_S2_BN2_B',    bn[3])

    # ── CNN Stage 3 ───────────────────────────────────────────────────────────
    print('\n[CNN Stage 3]')
    f.write('// CNN Stage 3\n')
    write_int8 (f, 'W_S3_C1',       get_conv_w('cnn.stage3.0'))
    bn = get_bn('cnn.stage3.1')
    write_float(f, 'W_S3_BN1_MEAN', bn[0])
    write_float(f, 'W_S3_BN1_VAR',  bn[1])
    write_float(f, 'W_S3_BN1_G',    bn[2])
    write_float(f, 'W_S3_BN1_B',    bn[3])
    write_int8 (f, 'W_S3_C2',       get_conv_w('cnn.stage3.3'))
    bn = get_bn('cnn.stage3.4')
    write_float(f, 'W_S3_BN2_MEAN', bn[0])
    write_float(f, 'W_S3_BN2_VAR',  bn[1])
    write_float(f, 'W_S3_BN2_G',    bn[2])
    write_float(f, 'W_S3_BN2_B',    bn[3])

    # ── Projection ────────────────────────────────────────────────────────────
    print('\n[Projection]')
    f.write('// Projection Linear(128->256)\n')
    write_int8 (f, 'W_PROJ',   get_linear_w('proj'))
    write_float(f, 'W_PROJ_B', get_linear_b('proj'))

    # ── CLS token + Positional embedding ──────────────────────────────────────
    print('\n[CLS + PosEmbed]')
    f.write('// CLS token and positional embedding\n')
    cls = model.cls_token.detach().squeeze().numpy().astype(np.float32)
    write_float(f, 'W_CLS_TOKEN', cls)
    pos = model.pos_embed.detach().squeeze(0).numpy().astype(np.float32)  # (153,256)
    # Write as flat array — indexing in C: W_POS_EMBED[t*256 + d]
    write_float(f, 'W_POS_EMBED', pos.flatten())

    # ── Transformer Blocks (flat naming, no structs) ───────────────────────────
    for i in range(3):
        print(f'\n[Transformer Block {i}]')
        f.write(f'// Transformer Block {i}\n')
        prefix = f'W_TFM{i}'

        write_float(f, f'{prefix}_LN1_G',  nm[f'blocks.{i}.norm1'].weight.detach().numpy())
        write_float(f, f'{prefix}_LN1_B',  nm[f'blocks.{i}.norm1'].bias.detach().numpy())
        write_float(f, f'{prefix}_LN2_G',  nm[f'blocks.{i}.norm2'].weight.detach().numpy())
        write_float(f, f'{prefix}_LN2_B',  nm[f'blocks.{i}.norm2'].bias.detach().numpy())

        write_int8 (f, f'{prefix}_QKV_W',  get_linear_w(f'blocks.{i}.attn.qkv'))
        write_float(f, f'{prefix}_QKV_B',  get_linear_b(f'blocks.{i}.attn.qkv'))
        write_int8 (f, f'{prefix}_OUT_W',  get_linear_w(f'blocks.{i}.attn.proj'))
        write_float(f, f'{prefix}_OUT_B',  get_linear_b(f'blocks.{i}.attn.proj'))

        params = dict(model.named_parameters())

        # Then inside the loop, replace the alpha line with:
        alpha = float(params[f'blocks.{i}.attn.alpha'].item())
        write_scalar(f, f'{prefix}_ALPHA', alpha)

        write_int8 (f, f'{prefix}_MLP1_W', get_linear_w(f'blocks.{i}.mlp.0'))
        write_float(f, f'{prefix}_MLP1_B', get_linear_b(f'blocks.{i}.mlp.0'))
        write_int8 (f, f'{prefix}_MLP2_W', get_linear_w(f'blocks.{i}.mlp.3'))
        write_float(f, f'{prefix}_MLP2_B', get_linear_b(f'blocks.{i}.mlp.3'))

    # ── Final LayerNorm ────────────────────────────────────────────────────────
    print('\n[Final LN + Head]')
    f.write('// Final LayerNorm\n')
    write_float(f, 'W_FINAL_LN_G', nm['norm'].weight.detach().numpy())
    write_float(f, 'W_FINAL_LN_B', nm['norm'].bias.detach().numpy())

    # ── Classifier head ────────────────────────────────────────────────────────
    f.write('// Head\n')
    write_int8 (f, 'W_HEAD1',   get_linear_w('head.0'))
    write_float(f, 'W_HEAD1_B', get_linear_b('head.0'))
    write_int8 (f, 'W_HEAD2',   get_linear_w('head.3'))
    write_float(f, 'W_HEAD2_B', get_linear_b('head.3'))

    f.write('#endif // WEIGHTS_H\n')

print(f'\nweights.h: {h_path.stat().st_size/1e6:.2f} MB')

# ── TEST SAMPLE ───────────────────────────────────────────────────────────────
print('\n[Test sample]')
if CACHE_PATH.exists():
    cache  = np.load(CACHE_PATH)
    specs  = cache['specs']
    labels = cache['labels']
    gmean  = float(cache['global_mean'])
    gstd   = float(cache['global_std'])

    idx       = np.where(labels == 0)[0][0]
    spec_norm = (specs[idx] - gmean) / (gstd + 1e-8)

    # Verify prediction
    with torch.no_grad():
        x      = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(x)
        pred   = int(logits.argmax(1).item())
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    CLASS_NAMES = ['Motorboat','Fishing','Cargo','Tugboat','Environment']
    lbl = int(labels[idx])
    print(f'  True: {lbl} ({CLASS_NAMES[lbl]})  Pred: {pred} ({CLASS_NAMES[pred]})  {"PASS" if pred==lbl else "FAIL"}')
    print(f'  Probs: {[f"{p:.3f}" for p in probs]}')

    spec_int8 = np.clip(np.round(spec_norm * 127.0), -128, 127).astype(np.int8)
    spec_int8.tofile(TB_DIR / 'test_mel.bin')
    np.array([lbl], dtype=np.int32).tofile(TB_DIR / 'test_label.bin')
    print(f'  test_mel.bin saved    ({spec_int8.nbytes} bytes)')
    print(f'  test_label.bin saved  (label={lbl})')
else:
    print(f'  SKIP: {CACHE_PATH} not found')

print('\n' + '='*60)
print('  DONE — all files written successfully')
print('='*60)