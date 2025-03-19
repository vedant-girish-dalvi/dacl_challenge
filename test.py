import numpy as np
import cv2

# mask = cv2.imread("C:\AInspect2Twin\dacl\dacl10k_v2_devphase\masks\train\dacl10k_v2_train_0000.png", cv2.IMREAD_GRAYSCALE)  # Load mask
# print(np.unique(mask))  # Print unique values in mask


mask = np.eye(19)
print(mask)[0]