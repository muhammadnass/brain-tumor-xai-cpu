#!/usr/bin/env python3
"""
Standalone test of the PrunedUNet3D construction used by
scripts/prune_bottleneck.py.

Reproduces the project's UNet3D/ConvBlock3D definitions verbatim, builds a
narrow instance so the test runs in seconds, drives its bottleneck into the
collapsed state observed in the trained checkpoints (BatchNorm gamma and
beta at zero), and then asserts that PrunedUNet3D reproduces the
zero-ablated network bit-for-bit.

This is deliberately self-contained: it does not import the project, does
not touch a checkpoint, and does not need the dataset. Its only job is to
show that the pruning construction is sound before it is applied to a real
checkpoint.

Run:  python3 tests/test_prune_logic.py
Exit: 0 if every check passes, 1 otherwise.
"""
import copy
import sys
from typing import Tuple

import torch
import torch.nn as nn


# ---- verbatim from src/ai_models/segmentation.py -------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3,
                 features: Tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for i in range(len(features)):
            in_ch = in_channels if i == 0 else features[i - 1]
            self.encoder.append(ConvBlock3D(in_ch, features[i]))
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for i in range(len(features) - 1, -1, -1):
            up_in = features[i] * 2 if i == len(features) - 1 else features[i] * 2
            up_out = features[i]
            self.upconv.append(
                nn.ConvTranspose3d(up_in, up_out, kernel_size=2, stride=2))
            dec_in = features[i] * 2 if i < len(features) - 1 else up_out + features[i]
            self.decoder.append(ConvBlock3D(dec_in, features[i]))
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(len(self.decoder)):
            x = self.upconv[i](x)
            skip = encoder_outputs[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)
        return self.final_conv(x)


# ---- the construction under test -----------------------------------
class PrunedUNet3D(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.features = tuple(src.features)
        self.encoder = copy.deepcopy(src.encoder)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.decoder = copy.deepcopy(src.decoder)
        self.upconv = nn.ModuleList(
            copy.deepcopy(m) for m in list(src.upconv)[1:])
        self.final_conv = copy.deepcopy(src.final_conv)
        up0 = src.upconv[0]
        if up0.bias is None:
            raise RuntimeError("upconv[0] has no bias")
        with torch.no_grad():
            self.register_buffer("up0_const", up0.bias.detach().clone())
        self.eval()

    def forward(self, x):
        enc = []
        for stage in self.encoder:
            x = stage(x)
            enc.append(x)
            x = self.pool(x)
        skip = enc[-1]
        x = self.up0_const.view(1, -1, 1, 1, 1).expand(
            skip.shape[0], -1, *skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        x = self.decoder[0](x)
        for i in range(1, len(self.decoder)):
            x = self.upconv[i - 1](x)
            x = torch.cat([x, enc[-(i + 1)]], dim=1)
            x = self.decoder[i](x)
        return self.final_conv(x)


def collapse_bottleneck(net):
    """Drive the bottleneck into the state seen in the trained checkpoints:
    BatchNorm affine parameters at zero, so the block emits zero whatever
    its input, and the trailing ReLU keeps it there."""
    with torch.no_grad():
        for mod in net.bottleneck.conv:
            if isinstance(mod, nn.BatchNorm3d):
                mod.weight.zero_()
                mod.bias.zero_()
                mod.running_var.fill_(5.605e-45)
                mod.running_mean.zero_()


def main():
    torch.manual_seed(0)
    feats = (2, 4, 8, 16, 32)          # narrow, same depth as the real net
    net = UNet3D(features=feats)
    net.eval()
    collapse_bottleneck(net)

    # confirm the bottleneck really is inert in this fixture
    x0 = torch.randn(1, 4, 32, 32, 32)
    captured = {}
    h = net.bottleneck.register_forward_hook(
        lambda m, i, o: captured.update(out=o.detach()))
    with torch.no_grad():
        net(x0)
    h.remove()
    bmax = float(captured["out"].abs().max())
    print("fixture bottleneck max|activation| = %.3e" % bmax)
    assert bmax == 0.0, "fixture is not inert; the test would prove nothing"

    pruned = PrunedUNet3D(net)

    n_total = sum(p.numel() for p in net.parameters())
    n_pruned = sum(p.numel() for p in pruned.parameters())
    n_bn = sum(p.numel() for p in net.bottleneck.parameters())
    n_up0 = sum(p.numel() for p in net.upconv[0].parameters())
    print("params: total %d | bottleneck %d | upconv0 %d | pruned %d "
          "| removed %d" % (n_total, n_bn, n_up0, n_pruned,
                            n_total - n_pruned))
    assert n_total - n_pruned == n_bn + n_up0, \
        "removed parameter count does not equal bottleneck + upconv[0]"

    ok = True
    for trial in range(5):
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            li = net(x)
            h = net.bottleneck.register_forward_hook(
                lambda m, i, o: torch.zeros_like(o))
            la = net(x)
            h.remove()
            lp = pruned(x)
        bit_abl = torch.equal(lp, la)
        bit_int = torch.equal(lp, li)
        d_abl = float((lp - la).abs().max())
        d_int = float((lp - li).abs().max())
        md_abl = int((lp.argmax(1) != la.argmax(1)).sum())
        md_int = int((lp.argmax(1) != li.argmax(1)).sum())
        print("trial %d: vs ablated bitwise=%s max|d|=%.3e maskdiff=%d | "
              "vs intact bitwise=%s max|d|=%.3e maskdiff=%d"
              % (trial, bit_abl, d_abl, md_abl, bit_int, d_int, md_int))
        ok &= bit_abl and md_abl == 0 and md_int == 0

    # batch dimension > 1 must also work (the constant is broadcast)
    xb = torch.randn(3, 4, 32, 32, 32)
    with torch.no_grad():
        h = net.bottleneck.register_forward_hook(
            lambda m, i, o: torch.zeros_like(o))
        lab = net(xb)
        h.remove()
        lpb = pruned(xb)
    bit_b = torch.equal(lpb, lab)
    print("batch=3: vs ablated bitwise=%s max|d|=%.3e"
          % (bit_b, float((lpb - lab).abs().max())))
    ok &= bit_b

    # state_dict must round-trip
    sd = pruned.state_dict()
    rebuilt = PrunedUNet3D(net)
    rebuilt.load_state_dict(sd, strict=True)
    rebuilt.eval()
    with torch.no_grad():
        same = torch.equal(rebuilt(x0), pruned(x0))
    print("state_dict round-trip reproduces output: %s" % same)
    ok &= same

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
