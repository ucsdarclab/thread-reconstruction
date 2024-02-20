import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

def SAM_segmentation(image):
    sam_checkpoint = "/Users/neelay/Downloads/segmentation/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    print("Model found")

    predictor = SamPredictor(sam)

    print("Generating embedding")

    predictor.set_image(image)

    print("Embedding generated!")

    points = []
    labels = []
    masks = None
    w_name = "Add (L-click) and Remove (R-click) Mask"

    # select points, press any key to end selection
    def select_point(event, x, y, flags, param):
        nonlocal points, labels, masks
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN: 
            points.append((x, y))
            labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            masks, _, _ = predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                box=None,
                multimask_output=False,
            )
            clone = image.copy()
            clone = cv2.cvtColor(clone, cv2.COLOR_RGB2BGR)
            clone[masks[0]>0] = np.array([255, 144, 33])
            for point, label in zip(points, labels):
                cv2.circle(clone, point, 5, 
                           (0,255,0) if label==1 else (0,0,255), 2)
            cv2.imshow(w_name, clone)
        
    cv2.namedWindow(w_name)
    cv2.setMouseCallback(w_name, select_point)
    cv2.imshow(w_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return masks[0]*255
