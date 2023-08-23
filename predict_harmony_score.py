import torch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.bargainnet_model import StyleEncoder
import torch.nn.functional as F
from torch import nn
import functools
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import math
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description='Harmony Predictor')
    parser.add_argument('--weight', default='checkpoints/latest_net_E.pth', type=str, help='path to the weight file of net_E')
    parser.add_argument('--image', default='examples/composite/example_1.png', type=str, help='composite image')
    parser.add_argument('--mask', default='examples/mask/example_1.png', type=str, help='foreground mask')
    parser.add_argument('--gpu', default=0, type=int, help='device ID')
    args = parser.parse_args()
    return args

class InharmonyLevelPredictor:
    def __init__(self, device, weight_path):
        self.device = device
        self.model  = self.build_inharmony_predictor(weight_path)
        image_size  = 256
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def build_inharmony_predictor(self, weight_path):
        model = StyleEncoder(style_dim=16)
        assert os.path.exists(weight_path), weight_path
        print('Build HarmonyLevel Predictor')
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model = model.eval().to(self.device)
        return model

    def data_preprocess(self, image, mask):
        # normalize foreground mask
        fg_mask = mask
        bg_mask = (255 - mask.astype(np.float32)).astype(np.uint8)
        fg_mask = self.mask_transform(Image.fromarray(fg_mask))
        fg_mask = fg_mask.unsqueeze(0).to(self.device)
        bg_mask = self.mask_transform(Image.fromarray(bg_mask))
        bg_mask = bg_mask.unsqueeze(0).to(self.device)

        image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image   = self.transform(Image.fromarray(image))
        image   = image.unsqueeze(0).to(self.device)
        return image, bg_mask, fg_mask

    def Euclidean_distance(self, vec1, vec2):
        vec1 = vec1.cpu().numpy()
        vec2 = vec2.cpu().numpy()
        dist = np.sqrt(np.sum((vec1 - vec2)**2))
        return dist

    def __call__(self, image, mask):
        with torch.no_grad():
            im, bg_mask, fg_mask = self.data_preprocess(image, mask)
            bg_sty_vector = self.model(im, bg_mask)
            fg_sty_vector = self.model(im, fg_mask)
        eucl_dist = self.Euclidean_distance(bg_sty_vector, fg_sty_vector)
        # convert distance to harmony level which lies in 0 and 1
        harm_level = math.exp(-0.04212 * eucl_dist)
        harm_level = round(harm_level, 2)
        return harm_level
    
if __name__ == '__main__':
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}')
    predictor = InharmonyLevelPredictor(device, args.weight)
    assert os.path.exists(args.image), args.image
    assert os.path.exists(args.mask), args.mask
    img  = cv2.imread(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    harm_score = predictor(img, mask)
    print('The harmony score of {} is {:.2f}'.format(os.path.basename(args.image), harm_score))