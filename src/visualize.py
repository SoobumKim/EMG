import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def visual_jpeg(emg, age):
    for idx, (emg_i, age_i) in enumerate(zip(emg, age)):
        save_dir = "emg_images"
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(emg_i, torch.Tensor):
            emg_i = emg_i.cpu().numpy()
        if isinstance(age_i, torch.Tensor):
            age_i = age_i.item()  # age가 tensor인 경우 float로

        # 특징 추출
        rms = np.sqrt(np.mean(emg_i**2))
        mav = np.mean(np.abs(emg_i))
        zc = np.sum(np.diff(np.sign(emg_i)) != 0)
        wl = np.sum(np.abs(np.diff(emg_i)))

        # 시각화
        plt.figure(figsize=(6, 3))
        plt.plot(emg_i, linewidth=1)
        plt.title(
            f"Age: {age_i} | RMS: {rms:.2f}, MAV: {mav:.2f}, ZC: {zc}, WL: {wl:.1f}",
            fontsize=9,
        )
        plt.axis("off")  # 축 제거

        # 저장
        save_path = os.path.join(save_dir, f"{idx}_{age_i}.jpg")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
