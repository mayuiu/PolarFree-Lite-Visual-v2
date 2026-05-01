import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image


DATA_DIR = r"D:\APP\作业\吉海萍_毕设\1\data"

# 请在这里填入5组训练数据的路径，每组5张 [0, 45, 90, 135, Label]
TRAIN_GROUPS = [
    ["0001_000.png", "0001_045.png", "0001_090.png", "0001_0135.png", "0000_000.png"],  # 第一组
    ["p0_2.png", "p45_2.png", "p90_2.png", "p135_2.png", "gt_2.png"],  # 第二组
    ["p0_3.png", "p45_3.png", "p90_3.png", "p135_3.png", "gt_3.png"],  # 第三组
    ["p0_4.png", "p45_4.png", "p90_4.png", "p135_4.png", "gt_4.png"],  # 第四组
    ["p0_5.png", "p45_5.png", "p90_5.png", "p135_5.png", "gt_5.png"],  # 第五组
]

# 数据目录（你的数据根目录）
DATA_ROOT = r"D:\APP\作业\吉海萍_毕设\1\data"

# 待处理的 4 张偏振图（脚本会读取以下 1.jpg,2.jpg,3.jpg,4.jpg）
TEST_INPUT_DIR = DATA_ROOT
TEST_INPUT_FILES = [os.path.join(TEST_INPUT_DIR, f"{i}.jpg") for i in [1,2,3,4]]

# 输出目录
OUT_DIR = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Training params (小规模训练)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = 600  # 小样本多轮训练
BATCH_SIZE = 1
IMG_SIZE = 256  # resize 为 256x256，可调

# ----------------------------
# Utility I/O
def load_rgb(path, resize=None):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    im = im[:, :, ::-1]  # BGR->RGB
    if resize:
        im = cv2.resize(im, (resize, resize), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    return im

def save_uint8(path, img):
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img[:, :, ::-1])  # RGB->BGR

# ----------------------------
# Physical model functions (快速的偏振差分 + 菲涅尔近似融合)
def simple_polar_diff(I0, I45, I90, I135):
    # 根据偏振相位差，估计非偏振反射和偏振成分的分离
    # 使用 Stokes 简化方法： I = [I0, I45, I90, I135]
    # S0 = I0 + I90, S1 = I0 - I90, S2 = I45 - I135
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-6)
    # 估计偏振反射为 DoLP * S0 / 2 （经验式）
    polarized = (DoLP[..., None] * S0[..., None]) / 2.0
    diffuse = np.clip(S0[..., None]/2.0 - polarized, 0, 1)
    # diffuse 为去反射近似
    return diffuse, polarized

# ----------------------------
# Simple UNet-like model for refinement
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_c=4, out_c=3):
        super().__init__()
        self.enc1 = ConvBlock(in_c, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(128+64, 64)
        self.dec2 = ConvBlock(64+32, 32)
        self.final = nn.Conv2d(32, out_c, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = torch.sigmoid(self.final(d2))
        return out

# ----------------------------
# Dataset (从每组 5 张生成训练样本)
class PolarDataset(Dataset):
    def __init__(self, groups, img_size=256):
        self.samples = []
        self.size = img_size
        for g in groups:
            I0 = load_rgb(g[0], resize=img_size)
            I45 = load_rgb(g[1], resize=img_size)
            I90 = load_rgb(g[2], resize=img_size)
            I135 = load_rgb(g[3], resize=img_size)
            gt = load_rgb(g[4], resize=img_size)
            # build input: stack four polarization channels (as 3-channel each) into 12-ch?
            # 为简单起见，把四个角度三通道按通道拼接为 12 通道再压成 4 通道（平均每角保留灰度）
            # 这里改为将每角转换为灰度单通道 -> 4 通道输入
            def rgb2gray(im): return np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
            G0 = rgb2gray(I0)[..., None]
            G45 = rgb2gray(I45)[..., None]
            G90 = rgb2gray(I90)[..., None]
            G135 = rgb2gray(I135)[..., None]
            inp = np.concatenate([G0, G45, G90, G135], axis=2)  # HxWx4
            self.samples.append((inp.astype(np.float32), gt.astype(np.float32)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        inp, gt = self.samples[idx]
        inp = torch.from_numpy(inp.transpose(2,0,1))  # C,H,W
        gt = torch.from_numpy(gt.transpose(2,0,1))
        return inp, gt

# ----------------------------
# Training routine (small dataset)
def train_model(train_groups):
    dataset = PolarDataset(train_groups, img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = UNetSmall(in_c=4, out_c=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for inp, gt in loader:
            inp = inp.to(DEVICE)
            gt = gt.to(DEVICE)
            opt.zero_grad()
            out = model(inp)
            loss = loss_fn(out, gt)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, loss={total/len(loader):.6f}")
    return model

# ----------------------------
# Inference pipeline for a test scene (4 polar images -> outputs)
def inference_pipeline(model, I0, I45, I90, I135):
    # 1) Simple physical baseline
    diffuse_phy, polar_comp = simple_polar_diff(I0, I45, I90, I135)

    # 2) DL refinement: prepare 4-channel gray input
    def rgb2gray(im): return np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
    G0 = rgb2gray(I0)[..., None]; G45 = rgb2gray(I45)[..., None]
    G90 = rgb2gray(I90)[..., None]; G135 = rgb2gray(I135)[..., None]
    inp = np.concatenate([G0,G45,G90,G135], axis=2)
    inp_t = torch.from_numpy(inp.astype(np.float32).transpose(2,0,1))[None].to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(inp_t).cpu().numpy()[0].transpose(1,2,0)  # H,W,3

    # 3) Naive fusion baseline (mean of four RGB images)
    mean_rgb = (I0 + I45 + I90 + I135) / 4.0

    # 4) Polar-diff colorized: scale diffuse_phy to RGB by scaling per-channel from mean_rgb
    diffuse_rgb = np.clip(diffuse_phy * (mean_rgb / (np.mean(mean_rgb, axis=2, keepdims=True)+1e-6)), 0, 1)

    # 5) Combine DL result with physical estimate (residual learning): out is final refined
    refined = out  # network output already aims to be de-reflected RGB

    return {
        "original_mean": mean_rgb,
        "physical_diffuse": diffuse_rgb,
        "dl_refined": refined,
        "polar_component": polar_comp
    }

# ----------------------------
# Main flow
def main():
    if len(TRAIN_GROUPS) != 5:
        print("ERROR: 请在 TRAIN_GROUPS 中填写 5 组样本（每组5张）。脚本将在你的填写后运行。")
        return

    print("准备训练（小样本）...")
    model = train_model(TRAIN_GROUPS)
    # 保存模型
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "unet_refine.pth"))

    # 读取待处理的四张偏振图
    print("读取待处理图像...")
    imgs = []
    for p in TEST_INPUT_FILES:
        if not os.path.exists(p):
            print(f"ERROR: 测试输入文件不存在: {p}")
            return
        imgs.append(load_rgb(p, resize=IMG_SIZE))
    I0, I45, I90, I135 = imgs

    results = inference_pipeline(model, I0, I45, I90, I135)

    # 输出所有结果：原图(平均), 物理, DL, polar component
    save_uint8(os.path.join(OUT_DIR, "original_mean.png"), results["original_mean"])
    save_uint8(os.path.join(OUT_DIR, "physical_diffuse.png"), results["physical_diffuse"])
    save_uint8(os.path.join(OUT_DIR, "dl_refined.png"), results["dl_refined"])
    # polar component可视化（缩放）
    pol_vis = np.clip(results["polar_component"] / (np.max(results["polar_component"])+1e-6), 0, 1)
    save_uint8(os.path.join(OUT_DIR, "polar_component.png"), pol_vis)

    # 最终合并策略：把 DL refined 与 4 个角度融合为一张最终图（这里采用简单加权：DL为主，物理修正）
    final = results["dl_refined"] * 0.85 + results["physical_diffuse"] * 0.15
    save_uint8(os.path.join(OUT_DIR, "final_combined.png"), final)

    print("输出已保存到：", OUT_DIR)

if __name__ == "__main__":
    main()

