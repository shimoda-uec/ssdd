
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.vgg16_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", required=True, type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_la_crf", default=None, type=str)
    parser.add_argument("--out_ha_crf", default=None, type=str)
    parser.add_argument("--out_cam_pred", default=None, type=str)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        print('ss',len(cam_list),cam_list[0].shape, np.sum(cam_list[0:1], axis=0).shape, sum_cam.shape)
        print(cam_list[0],cam_list[1])
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0])*0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))
            cv2.imwrite('/export/space/shimoda-k/webly/lib/psa/res/cam_pred'+ str(iter)+ '.png', pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            print(bgcam_score.shape)
            print(bgcam_score)
            print(np.max(bgcam_score))
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]
                
            tmp=1-bg_score[0]
            print(orig_img.shape)
            fig = plt.figure()
            #ax = fig.add_axes([0, 0, 1, 1], frameon=False)
            #ax.set_xlim(0, 1), ax.set_xticks([])
            #ax.set_ylim(0, 1), ax.set_yticks([])
            plt.axis('off')  # 1
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)  # 2
            plt.imshow(tmp, cmap='jet')
            #canvas = FigureCanvas(fig)
            #canvas.draw()       # draw the canvas, cache the renderer
            #s, (width, height) = canvas.print_to_buffer()
            #width, height = fig.get_size_inches() * fig.get_dpi()
            #print(width,height)
            #image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
            #image = image[:,:,::-1]
            #print(image.shape)
            plt.savefig('./tmp.jpg', bbox_inches='tight', pad_inches=0)
            image=cv2.imread('./tmp.jpg')
            image=cv2.resize(image,(orig_img.shape[1],orig_img.shape[0]))
            bimage=orig_img*0.7+image*0.3
            #cv2.imwrite('./tmp.jpg',bimage.astype(np.uint8))
            return n_crf_al, bimage.astype(np.uint8)

        def get_palette(num_cls):
            """ Returns the color map for visualizing the segmentation mask.
            Args:
                num_cls: Number of classes
            Returns:
                The color map
            """
            n = num_cls
            palette = [0] * (n * 3)
            for j in range(0, n):
                lab = j
                palette[j * 3 + 0] = 0
                palette[j * 3 + 1] = 0
                palette[j * 3 + 2] = 0
                i = 0
                while lab:
                    palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                    palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                    palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                    i += 1
                    lab >>= 3
            return palette

        def mask2png(mask):
            palette = get_palette(256)
            mask=Image.fromarray(mask.astype(np.uint8))
            savedir='/export/space/shimoda-k/webly/lib/instance_segmentation/save_images/'
            mask.putpalette(palette)
            mask.save(savedir+'tp.png')
            mask=cv2.imread(savedir+'tp.png')
            return mask

        if args.out_la_crf is not None:
            crf_la, bimage = _crf_with_alpha(cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)
            keys=crf_la.keys()
            print(keys)
            for i, key in enumerate(keys):
              if i==0:
                cmap=crf_la[key]
                crf_map=np.zeros((len(keys),cmap.shape[0],cmap.shape[1]))
                crf_map[i]=cmap
              else:
                crf_map[i]=crf_la[key]
            crf_mask=np.argmax(crf_map,axis=0)
            mask=np.zeros_like(crf_mask)
            for i, key in enumerate(keys):
              loc=np.where(crf_mask==i)
              mask[loc]=key
            mask=mask2png(mask)
            cv2.imwrite('/export/space0/shimoda-k/wseg/psa/res/la_crf'+ str(iter)+ '.png', mask)
            cv2.imwrite('/export/space0/shimoda-k/wseg/psa/res/la_cam'+ str(iter)+ '.png', bimage)

        if args.out_ha_crf is not None:
            crf_ha, bimage = _crf_with_alpha(cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)
            keys=crf_ha.keys()
            print(keys)
            for i, key in enumerate(keys):
              if i==0:
                cmap=crf_ha[key]
                crf_map=np.zeros((len(keys),cmap.shape[0],cmap.shape[1]))
                crf_map[i]=cmap
              else:
                crf_map[i]=crf_ha[key]
            crf_mask=np.argmax(crf_map,axis=0)
            mask=np.zeros_like(crf_mask)
            for i, key in enumerate(keys):
              loc=np.where(crf_mask==i)
              mask[loc]=key
            mask=mask2png(mask)
            print('/export/space0/shimoda-k/wseg/psa/res/ha_crf'+ str(iter)+ '.png')
            cv2.imwrite('/export/space0/shimoda-k/wseg/psa/res/ha_crf'+ str(iter)+ '.png', mask)
            cv2.imwrite('/export/space0/shimoda-k/wseg/psa/res/ha_cam'+ str(iter)+ '.png', bimage)

        print(iter)

