import matplotlib

matplotlib.use("Agg")
import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys

sys.path.append(str(Path(__file__).absolute() / "../../common"))

# from config import cfg  # NOQA
from inter_model import get_model

# from utils.vis import vis_keypoints  # NOQA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--model_path", type=str, default="output/model_dump/snapshot_20.pth.tar"
    )
    parser.add_argument("imgs_path", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cudnn.benchmark = True
    device = args.gpu

    model_path = Path(args.model_path)
    assert model_path.is_file(), "Cannot find model at {}".format(model_path)
    print(("Load checkpoint from {}".format(model_path)))

    model = get_model("test", 21)
    ckpt = torch.load(str(model_path))
    model.load_state_dict(ckpt["network"], strict=False)
    model.eval()
    model.to(device)

    imgs_path = Path(args.imgs_path)
    with torch.no_grad():
        for idx, img_path in enumerate(imgs_path.glob("*.jpg")):
            print("{} {}".format(idx, img_path))

            # forward
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (256, 256))
            img_cpu = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            img_gpu = torch.from_numpy(img_cpu).unsqueeze(0).to(device)
            out = model(img_gpu)

            joint_coord_out = out["joint_coord"].cpu().numpy()
            print(joint_coord_out.shape)


if __name__ == "__main__":
    main()
