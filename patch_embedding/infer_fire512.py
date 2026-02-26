import torch
import torch.nn as nn
import time
import os
import copy
try: import psutil
except ImportError: psutil = None
from torchvision import models
try: from thop import profile
except ImportError: profile = None

class Fire(nn.Module):
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)

class Fire512(nn.Module):
    def __init__(self):
        super(Fire512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Fire(32, 8, 32, 32),
            Fire(64, 8, 32, 32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

def get_backbone(name):
    if name == "Fire512": return Fire512()
    elif name == "SqueezeNet": m = models.squeezenet1_1(weights=None); m.classifier = nn.Identity(); return m
    elif name == "ShuffleNet": m = models.shufflenet_v2_x1_0(weights=None); m.fc = nn.Identity(); return m
    elif name == "EfficientNet-B0": m = models.efficientnet_b0(weights=None); m.classifier = nn.Identity(); return m
    elif name == "MobileNetV3": m = models.mobilenet_v3_large(weights=None); m.classifier = nn.Identity(); return m
    elif name == "ResNet18": m = models.resnet18(weights=None); m.fc = nn.Identity(); return m
    elif name == "ResNet50": m = models.resnet50(weights=None); m.fc = nn.Identity(); return m

def benchmark():
    device_list = ["cpu"]
    if torch.cuda.is_available(): device_list.append("cuda")
    model_names = ["Fire512", "SqueezeNet", "ShuffleNet", "EfficientNet-B0", "MobileNetV3", "ResNet18", "ResNet50"]
    dummy_input = torch.randn(1, 3, 224, 224)
    process = psutil.Process(os.getpid()) if psutil else None
    h = f"{'Model':<16} | {'Params':>8} | {'FLOPs':>8} | {'Device':<6} | {'Infer':>9} | {'FPS':>7} | {'RAM+(MB)':>9} | {'VRAM(MB)':>9}"
    print(h)
    print("-" * len(h))
    for name in model_names:
        model = get_backbone(name)
        params = sum(p.numel() for p in model.parameters())
        if profile:
            macs, _ = profile(copy.deepcopy(model), inputs=(dummy_input,), verbose=False)
            flops = macs
        else: flops = 0
        fp32_size, int16_size, int8_size = params*4/(1024**2), params*2/(1024**2), params*1/(1024**2)
        for device in device_list:
            dev = torch.device(device)
            model.to(dev)
            inp = dummy_input.to(dev)
            if device == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
            ram_before = process.memory_info().rss / (1024**2) if process else 0
            for _ in range(10): model(inp)
            start = time.time()
            iters = 50
            for _ in range(iters): model(inp)
            if device == "cuda": torch.cuda.synchronize()
            elapsed = (time.time() - start) / iters * 1000
            fps = 1000 / elapsed
            ram_after = process.memory_info().rss / (1024**2) if process else 0
            ram_delta = max(0, ram_after - ram_before)
            vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
            energy = (elapsed / 1000.0) * (250 if device == "cuda" else 65)
            p_s, f_s, i_s = f"{params/1e6:.2f}M", f"{flops/1e9:.2f}G", f"{elapsed:.2f}ms"
            print(f"{name:<16} | {p_s:>8} | {f_s:>8} | {device:<6} | {i_s:>9} | {fps:>7.1f} | {ram_delta:>9.2f} | {vram:>9.2f}")
        print(f" > Quantized: FP32: {fp32_size:.2f}MB | INT16: {int16_size:.2f}MB | INT8: {int8_size:.2f}MB | Est. Energy: {energy:.4f}J")
        print("-" * len(h))

if __name__ == "__main__":
    benchmark()
