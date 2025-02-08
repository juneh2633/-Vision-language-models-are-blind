import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class ViTEncoder(torch.nn.Module):
    def __init__(self, pretrained=True, output_dim=512):
        super().__init__()
        self.model = models.vit_b_16(pretrained=pretrained)
        self.model.heads = torch.nn.Identity()
        self.feature_dim = self.model.hidden_dim
        self.fc = torch.nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):
        features = self.model(x)
        return self.fc(features)


base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, "./image/image_4_thickness_2.png")
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6482, 0.5635, 0.4257], std=[0.2171, 0.2315, 0.2546]
        ),
    ]
)
image_pil = Image.fromarray(image_rgb)
input_tensor = transform(image_pil).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ViTEncoder(pretrained=True, output_dim=512).to(device)
encoder.eval()

with torch.no_grad():
    features = encoder(input_tensor.to(device))

print("ViT 특징 벡터 크기:", features.shape)

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(
    hsv, lower_red2, upper_red2
)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


red_contours = find_contours(mask_red)
blue_contours = find_contours(mask_blue)


def find_intersections_vit(contours1, contours2, features):
    intersections = set()
    for cnt1 in contours1:
        for cnt2 in contours2:
            for p1 in cnt1:
                for p2 in cnt2:
                    x1, y1 = p1[0]
                    x2, y2 = p2[0]

                    feature_distance = torch.norm(
                        features[:, :256] - features[:, 256:], dim=1
                    ).item()

                    if abs(x1 - x2) < 5 and abs(y1 - y2) < 5 and feature_distance < 5:
                        intersections.add((x1, y1))

    return intersections


intersections = find_intersections_vit(red_contours, blue_contours, features)
intersection_count = len(intersections)
print("교점의 개수 (ViT 기반):", intersection_count)

for x, y in list(intersections):
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imwrite("/mnt/data/intersection_vit_output.png", image)
