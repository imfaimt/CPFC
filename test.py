import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse

from datasets import get_test_dataset, inverse_normalize, get_test_dataset_2
import cv2

# from model76 import copyfnac
from models.model import DINOSAURpp, Visual_Encoder
import math
from k_utils import get_color_mask_from_slot
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ezvsl_vggss', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    parser.add_argument("--dropout_img", type=float, default=0, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")

    # Distributed params
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', default=False)

    # slots params
    parser.add_argument('--num_slots', type=int, default=10)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--slot_att_iter', type=int, default=5)
    parser.add_argument('--query_opt', action='store_true', default=False)
    # === Data Related Parameters ===
    parser.add_argument('--resize_to', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--ISA', action="store_true")
    # parser.add_argument('--token_num', type=int, default=196)
    parser.add_argument('--encoder', type=str, default="dinov2-vitb-14",
                        choices=["dinov2-vitb-14", "dino-vitb-16", "dino-vitb-8", "sup-vitb-16"])

    args = parser.parse_args()

    set_remaining_args(args)

    return args


def set_remaining_args(args):

    args.patch_size = int(args.encoder.split("-")[-1])
    args.token_num = (args.resize_to[0] * args.resize_to[1]) // (args.patch_size ** 2)



def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Models

    # audio_visual_model = FNAC(args.tau, args.out_dim, args.dropout_img, args.dropout_aud)

    # audio_visual_model = copyfnac(args.tau, args.out_dim, args.dropout_img, args.dropout_aud, args.num_slots,
    #                  args.slot_dim, args.slot_att_iter, args.query_opt)

    vis_encoder = Visual_Encoder(args)

    audio_visual_model = DINOSAURpp(args)
   
    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)  # [B, 1, 7, 7]
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            audio_visual_model.cuda(args.gpu)
            object_saliency_model.cuda(args.gpu)
            vis_encoder.cuda(args.gpu)

            audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
            object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])
            vis_encoder = torch.nn.parallel.DistributedDataParallel(vis_encoder, device_ids=[args.gpu])

    audio_visual_model.cuda(args.gpu)
    object_saliency_model.cuda(args.gpu)
    vis_encoder.cuda(args.gpu)

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    if args.testset == 'flickr':
        testdataset = get_test_dataset(args)
    elif args.testset == 'k_flickr_5k':
        testdataset = get_test_dataset(args)
    else:
        testdataset = get_test_dataset_2(args)
    print(f"Loaded test dataset with {len(testdataset)} samples.")
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("Loaded dataloader.")

    # cIoU, auc = validate2(testdataloader, audio_visual_model, args)
    # print(f'cIoU (epoch ): {cIoU}')
    # print(f'AUC (epoch ): {auc}')

    # validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args)
    # validate2(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args)

    validate(testdataloader, audio_visual_model, vis_encoder, object_saliency_model, viz_dir, args)
    validate2(testdataloader, audio_visual_model,vis_encoder,object_saliency_model, viz_dir, args)

@torch.no_grad()
def validate(testdataloader, audio_visual_model, vis_encoder,object_saliency_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)
    vis_encoder.train(False)

    evaluator_av = utils.Evaluator()
    evaluator_obj = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    for step, (image, spec, bboxes, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        
        # print( image.get_device(), spec.get_device())
        # Compute S_AVL
        features = vis_encoder(image.float())
        heatmap_av = audio_visual_model(features, spec.float())[1].unsqueeze(1)
        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()

        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        # Computer masks
        masks = audio_visual_model(features, spec.float())[4]
        B, S, token = masks.shape
        H = W = int(math.sqrt(token))
        masks_spatial = masks.view(B, S, H, W)

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_av_obj = utils.normalize_img(pred_av * args.alpha + pred_obj * (1 - args.alpha))

            gt_map = bboxes['gt_map'].data.cpu().numpy()

            thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
            evaluator_av.cal_CIOU(pred_av, gt_map, thr_av)

            thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * 0.5)]
            evaluator_obj.cal_CIOU(pred_obj, gt_map, thr_obj)

            thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
            evaluator_av_obj.cal_CIOU(pred_av_obj, gt_map, thr_av_obj)

            if args.save_visualizations:
                denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                denorm_image = (denorm_image*255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_image.jpg'), denorm_image)

                # visualize bboxes on raw images
                gt_boxes_img = utils.visualize(denorm_image, bboxes['bboxes'], test_set=args.testset)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_gt_boxes.jpg'), gt_boxes_img)

                # visualize heatmaps
                heatmap_img = np.uint8(pred_av*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av.jpg'), fin)

                heatmap_img = np.uint8(pred_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_obj.jpg'), fin)

                heatmap_img = np.uint8(pred_av_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av_obj.jpg'), fin)

                masks_upsampled = F.interpolate(masks_spatial, size=(224, 224), mode='bilinear', align_corners=False)
                slot_mask_segmap = get_color_mask_from_slot(masks_upsampled, 0, bs_idx=i).astype(np.uint8)
                fin = cv2.addWeighted(slot_mask_segmap, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_masks.jpg'), fin)

        # print(f'{step+1}/{len(testdataloader)}: map_av={evaluator_av.finalize_AP50():.2f} map_obj={evaluator_obj.finalize_AP50():.2f} map_av_obj={evaluator_av_obj.finalize_AP50():.2f}')

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc

    print('AV: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av)))
    print('Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_obj)))
    print('AV_Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av_obj)))

    utils.save_iou(evaluator_av.ciou, 'av', viz_dir)
    utils.save_iou(evaluator_obj.ciou, 'obj', viz_dir)
    utils.save_iou(evaluator_av_obj.ciou, 'av_obj', viz_dir)

# we test different interpolation settings
@torch.no_grad()
def validate2(testdataloader, audio_visual_model, vis_encoder,object_saliency_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)
    vis_encoder.train(False)

    evaluator_av = utils.Evaluator()
    evaluator_obj = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    for step, (image, spec, bboxes, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        
        # print( image.get_device(), spec.get_device())
        # Compute S_AVL
        features = vis_encoder(image.float())
        heatmap_av = audio_visual_model(features, spec.float())[1].unsqueeze(1)
        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_av = heatmap_av.data.cpu().numpy()

        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()




        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_av_obj = utils.normalize_img(pred_av * args.alpha + pred_obj * (1 - args.alpha))

            gt_map = bboxes['gt_map'].data.cpu().numpy()

            thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
            evaluator_av.cal_CIOU(pred_av, gt_map, thr_av)

            thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * 0.5)]
            evaluator_obj.cal_CIOU(pred_obj, gt_map, thr_obj)

            thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
            evaluator_av_obj.cal_CIOU(pred_av_obj, gt_map, thr_av_obj)

            if args.save_visualizations:
                denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                denorm_image = (denorm_image*255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_image.jpg'), denorm_image)

                # visualize bboxes on raw images
                gt_boxes_img = utils.visualize(denorm_image, bboxes['bboxes'], test_set=args.testset)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_gt_boxes.jpg'), gt_boxes_img)

                # visualize heatmaps
                heatmap_img = np.uint8(pred_av*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av.jpg'), fin)

                heatmap_img = np.uint8(pred_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_obj.jpg'), fin)

                heatmap_img = np.uint8(pred_av_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.5, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av_obj.jpg'), fin)

        # print(f'{step+1}/{len(testdataloader)}: map_av={evaluator_av.finalize_AP50():.2f} map_obj={evaluator_obj.finalize_AP50():.2f} map_av_obj={evaluator_av_obj.finalize_AP50():.2f}')

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc

    print('AV: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av)))
    print('Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_obj)))
    print('AV_Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av_obj)))

    utils.save_iou(evaluator_av.ciou, 'av', viz_dir)
    utils.save_iou(evaluator_obj.ciou, 'obj', viz_dir)
    utils.save_iou(evaluator_av_obj.ciou, 'av_obj', viz_dir)


class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


if __name__ == "__main__":
    main(get_arguments())

