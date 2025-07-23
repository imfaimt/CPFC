import os
import time
import math
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import json

from models.model import DINOSAURpp, Visual_Encoder

from read_args import get_args, print_args
import utils
from k_utils import save_slot_visualizations

from datasets import get_train_dataset, get_test_dataset, get_train_dataset_2, get_test_dataset_2, get_all_classes



def train_epoch(args, vis_encoder, model, optimizer, scheduler, train_loader, total_iter, writer,epoch):

    total_loss = 0.0
    vis_encoder.eval()
    model.train()

    loader = tqdm(train_loader, disable=(args.gpu != 0))

    for i, (frames, spec, _, file_id) in enumerate(loader):
        frames = frames.cuda(non_blocking=True)  # (B, 3, H, W)
        spec = spec.cuda(args.gpu, non_blocking=True)
        B = frames.shape[0]

        features = vis_encoder(frames.float())  # (B, token, 768)
        loss_list, out_map, reconstruction, slot, masks, aud = model(features ,spec.float(), file_id)  # (B, token, 768), (B, S, D_slot), (B, S, token)
        loss_recon = F.mse_loss(reconstruction, features.detach())

        if epoch < args.warmup:
            loss = loss_recon
        else:
            loss = 10 * loss_recon + args.loss1_weight * loss_list[1] + args.loss2_weight * loss_list[2]

            # loss = 10 * loss_recon + args.loss1_weight * loss_list[1] + loss_list[3]
            # [loss_recon, loss_aligned_slot, loss3, triplet_loss]
        total_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.gpu == 0:
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            mean_loss = total_loss / (i + 1)
            loader.set_description(
                f"lr: {lr:.6f} | mean_loss: {mean_loss:.5f} | loss: {loss.item():.5f}  | recon: {loss_recon.item():.5f}  | loss[0]: {loss_list[0].item():.5f} "
                f" | loss[1]: {loss_list[1]:.5f} | loss[2]: {loss_list[2]:.5f} | loss[3]: {loss_list[3]:.5f}")

            writer.add_scalar("batch/loss", loss.item(), total_iter)

            # 修改后的可视化保存
            if (i + 1) % 500 == 0:
                save_slot_visualizations(args, frames, masks,
                                         epoch=total_iter // len(train_loader),
                                         iter=i , file_id = file_id)

        total_iter += 1

    mean_loss = total_loss / (i + 1)
    return mean_loss, total_iter

@torch.no_grad()
def validate(test_loader, vis_encoder, model, args):

    vis_encoder.eval()
    model.eval()
    loader = tqdm(test_loader)

    evaluator = utils.Evaluator()

    for i, (image, spec, bboxes, _) in enumerate(loader):

        image = image.cuda(args.gpu, non_blocking=True)
        spec = spec.cuda(args.gpu, non_blocking=True)

        features = vis_encoder(image.float())  # (B, token, 768)
        avl_map = model(features, spec.float())[1].unsqueeze(1)
        avl_map = F.interpolate(avl_map, size=(224, 224), mode='bicubic', align_corners=False)
        avl_map = avl_map.data.cpu().numpy()

        for i in range(spec.shape[0]):
            pred = utils.normalize_img(avl_map[i, 0])
            gt_map = bboxes['gt_map'].data.cpu().numpy()
            thr = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            evaluator.cal_CIOU(pred, gt_map, thr)

    cIoU = evaluator.finalize_AP50()
    AUC = evaluator.finalize_AUC()
    return cIoU, AUC

def main_worker(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print_args(args)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # Get dataset
    traindataset = get_train_dataset(args)
    print(f"Dataset size: {len(traindataset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset, shuffle=True, num_replicas=args.gpus,
                                                                    rank=args.gpu)

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=False,sampler=train_sampler, drop_last=True,
                                               persistent_workers=args.workers > 0)

    print('train sample amount:', len(train_loader))
    if args.testset == 'flickr':
        testdataset = get_test_dataset(args)
    else:
        testdataset = get_test_dataset_2(args)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,persistent_workers=args.workers > 0)
    print('test sample amount:', len(test_loader))
    print("Loaded dataloader.")

    # === Model ===
    vis_encoder = Visual_Encoder(args).cuda()
    model = DINOSAURpp(args).cuda()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Training Items ===
    optimizer = torch.optim.Adam(utils.get_params_groups(model), lr=args.learning_rate)
    scheduler = utils.get_scheduler(args, optimizer, train_loader)

    # Resume if possible
    start_epoch, best_cIoU, best_Auc = 0, 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'best.pth'), map_location='cpu')
        start_epoch, best_cIoU, best_Auc = ckp['epoch'], ckp['best_cIoU'], ckp['best_Auc']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    # Training loop
    cIoU, auc = validate(test_loader, vis_encoder, model, args)
    print(f'cIoU (epoch {start_epoch}): {cIoU}')
    print(f'AUC (epoch {start_epoch}): {auc}')
    print(f'best_cIoU: {best_cIoU}')
    print(f'best_Auc: {best_Auc}')

    writer = utils.get_writer(args) if args.gpu == 0 else None
    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}
    if args.use_checkpoint:
        utils.restart_from_checkpoint(args,
                                      run_variables=to_restore,
                                      model=model,
                                      optimizer=optimizer,
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]
    start_time = time.time()

    dist.barrier()

    print("Starting training!")
    total_iter = 0
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        mean_loss, total_iter = train_epoch(args, vis_encoder, model, optimizer, scheduler, train_loader,
                                            total_iter, writer, epoch)
        # === Validate ===
        cIoU, auc = validate(test_loader, vis_encoder, model, args)
        print(f'cIoU (epoch {epoch + 1}): {cIoU}');
        print(f'AUC (epoch {epoch + 1}): {auc}')
        print(f'best_cIoU: {best_cIoU}');
        print(f'best_Auc: {best_Auc}')

        # === Save Checkpoint ===
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "best_cIoU": best_cIoU,
            "best_Auc": best_Auc
        }
        model_path = Path(model_dir) / 'latest.pth'
        utils.save_on_master(save_dict, model_path)
        print(f"Model saved to {model_dir}")
        if cIoU >= best_cIoU:
            best_cIoU, best_Auc = cIoU, auc
            model_path = Path(model_dir) / 'best.pth'
            torch.save(save_dict, model_path)

        # === Log ===
        if writer is not None:
            writer.add_scalar('loss/epoch', mean_loss, epoch + 1)
            writer.add_scalar('cIoU/epoch', cIoU, epoch + 1)
            writer.add_scalar('AUC/epoch', auc, epoch + 1)
            writer.flush()

        dist.barrier()

        print("===== ===== ===== ===== =====\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 在训练结束时关闭writer
    if writer is not None:
        writer.close()

    dist.destroy_process_group()



if __name__ == '__main__':
    args = get_args()
    main_worker(args)
