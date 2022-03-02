import torch
import torch.nn as nn
from torchvision.io import read_image
import numpy as np
import cv2


class Inference_Pipeline:
    def __init__(self, model, pred_size, aug):
        self.model = model 
        self.aug = aug
        self.pred_size = pred_size
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, batch, mode1="bilinear", mode2="nearest"):
        """
        input 4D tensor (B, C, H, W)
        output 3D tensor (B, 1, H, W)
        """
        _, _, H, W = batch.size()
        batch = nn.functional.interpolate(batch.float(), size=self.pred_size, mode=mode1)
        batch, _ = self.aug(batch, None)
        preds = self.model(batch).argmax(dim=1, keepdim=True)
        preds = nn.functional.interpolate(preds.float(), size=(H, W), mode=mode2).type(torch.uint8)
        return preds

    def class_to_color(self, class_img, color_dict):
        """
        input 3D array (H, W, 3)
        """
        assert len(class_img.shape) == 3, "class_img must have dimension 3."
        res = np.zeros(shape=class_img.shape)
        for key in color_dict: 
            res += (class_img == key) * color_dict[key]
        return res


def predict_save_image(source, dst, ss_pipeline, device):
    """
    source: image path
    dst: path where the prediction image is saved
    ss_pipeline: Inference_Pipline object 
    """
    img = read_image(source).unsqueeze(dim=0).to(device)
    out = ss_pipeline.predict(img).squeeze().to("cpu").numpy()
    cv2.imwrite(dst ,np.stack((out,)*3, axis=-1))


if __name__ == "__main__":

    H, W = 288, 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    test_aug = Test_Augmentation(...)
    model = torch.load("model.pth").to(device=DEVICE)

    ss_pipeline = Inference_Pipeline(model, pred_size=(H, W), aug = test_aug)

    for img_path in tqdm(glob.glob(source + "*.jpg")):
        save_path = os.path.join(dst, os.path.basename(img_path).split(".")[0] + ".png")
        predict_save_image(img_path, save_path, ss_pipeline) 