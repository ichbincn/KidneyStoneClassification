import torch
from torchvision.transforms import Resize
import scipy
import numpy as np
import os
import random
from skimage.transform import resize

def load_npy_data(data_dir):
    datanp = []  # images
    truenp = []  # labels
    for file in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        #         data[0][0] = resize(data[0][0], (224,224,224))
        #if (split == 'train'):
           # data_sug = transform.rotate(data[0][0], 60)  # 旋转60度，不改变大小
           # data_sug2 = exposure.exposure.adjust_gamma(data[0][0], gamma=0.5)  # 变亮
           # datanp.append(data_sug)
           # truenp.append(data[0][1])
           # datanp.append(data_sug2)
           # truenp.append(data[0][1])
        datanp.append(data[0][0])
        truenp.append(data[0][1])
    datanp = np.array(datanp)
    # numpy.array可使用 shape。list不能使用shape。可以使用np.array(list A)进行转换。
    # 不能随意加维度
    datanp = np.expand_dims(datanp, axis=4)  # 加维度,from(1256,256,128)to(256,256,128,1),according the cnn tabel.png
    datanp = datanp.transpose(0, 4, 1, 2, 3)
    truenp = np.array(truenp).reshape(-1, 4)
    print(datanp.shape, truenp.shape)
    print(np.min(datanp), np.max(datanp), np.mean(datanp), np.median(datanp))
    return datanp, truenp

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(12) + worker_id)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_patch_mask(img, pred_mask):
    oi_seg = torch.mul(img, pred_mask)
    zoom_seg = torch.cat([img, pred_mask], dim=1)
    return oi_seg, zoom_seg


def returnCAM(feature_conv, weight_softmax, idx, size=(256, 256, 256)):
    bz, nc, h, w, d = feature_conv.shape
    feature_conv = feature_conv.reshape((bz, nc, h * w * d))
    #zoom = Resize([256, 256, 128])
    cams = []
    for i in range(bz):
        cam_bi = torch.matmul(weight_softmax[idx[i], :], feature_conv[i, :, :])
        cam_bi = cam_bi.reshape((1, h, w, d))

        cam_img = (cam_bi - cam_bi.min()) / (cam_bi.max() - cam_bi.min())  # normalize
        cam_img = resize(cam_img, size)
        cam_img = torch.Tensor(cam_img)
        cam_img = cam_img.unsqueeze(0)

        cams.append(cam_img)
    out = torch.cat(cams, dim=0)

    return out