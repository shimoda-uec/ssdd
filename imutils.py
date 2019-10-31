import PIL.Image
import random
import numpy as np
import cv2

class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, inputs):
        if bool(random.getrandbits(1)):
            outputs=[]
            for inp in inputs:
                out = np.fliplr(inp).copy()
                outputs.append(out)
            return outputs
        else:
            return inputs


class RandomResizeLong():
    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long
    def __call__(self, inputs):
        img=inputs[0]
        target_long = random.randint(self.min_long, self.max_long)
        #w, h = img.size
        h, w, c = img.shape
        target_shape = (target_long, target_long)
        """
        if w > h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))
        """
        outputs=[]
        for inp in inputs:
            out = cv2.resize(inp, target_shape)
            if len(out.shape)==2:
                out=np.expand_dims(out,2)
            outputs.append(out)
        return outputs

class RandomCrop():
    def __init__(self, cropsize):
        self.cropsize = cropsize
    def __call__(self, inputs):
        imgarr = np.concatenate(inputs, axis=-1)
        h, w, c = imgarr.shape
        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)
        w_space = w - self.cropsize
        h_space = h - self.cropsize
        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0
        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0
            
        outputs=[]
        for inp in inputs:
            container = np.zeros((self.cropsize, self.cropsize, inp.shape[-1]), np.float32)
            container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
                inp[img_top:img_top+ch, img_left:img_left+cw]
            outputs.append(container)
        return outputs

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)
        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
        return proc_img


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

class Rescale():
    def __init__(self, scale):
        self.scale=scale
    def __call__(self, inputs):
        outputs=[]
        for inp in inputs:
            out = cv2.resize(inp, self.scale)
            if len(out.shape)==2:
                out=np.expand_dims(out,2)
            outputs.append(out)
        return outputs


def crf_inference(img, probs, t=3, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    h, w = img.shape[:2]
    n_labels = labels
    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)
    return np.array(Q).reshape((n_labels, h, w))