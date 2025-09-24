import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model for people counting
yolo_model = YOLO("yolov8n.pt")  # Using a pre-trained YOLOv8 model

# Define CSRNet model (same as trained model)
class CSRNet(torch.nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.backend = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# Load trained CSRNet model
csrnet = CSRNet().cuda()
csrnet.load_state_dict(torch.load("csrnet_crowd.pth"))
csrnet.eval()

# Process video feed
cap = cv2.VideoCapture("highcrowd_vid.mp4")  # Change to CCTV feed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO People Counting
    results = yolo_model(frame)
    people_count = sum(1 for obj in results[0].boxes if obj.cls == 0 and obj.conf > 0.5)

    # CSRNet Density Estimation
    input_frame = cv2.resize(frame, (256, 256))
    input_frame = torch.tensor(input_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

    with torch.no_grad():
        density_map = csrnet(input_frame).cpu().squeeze().numpy()

    density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
    people_count_csrnet = int(np.round(np.sum(density_map) * 1.25))
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

    # Display results
    # Final People Count: Use max of YOLO & CSRNet estimates
    people_count_final = max(people_count, people_count_csrnet)

    cv2.putText(frame, f"People Count: {people_count_final}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined = cv2.addWeighted(frame, 0.6, density_map, 0.4, 0)
    cv2.imshow("Crowd Management", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
