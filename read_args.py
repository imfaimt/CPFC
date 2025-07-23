import sys
import argparse

import torch

def set_remaining_args(args):
    args.gpus = torch.cuda.device_count()

    args.patch_size = int(args.encoder.split("-")[-1])
    args.token_num = (args.resize_to[0] * args.resize_to[1]) // (args.patch_size ** 2)

def print_args(args):
    print("====== Arguments ======")
    print(f"training name: {args.experiment_name}\n")

    print(f"dataset: {args.trainset}")
    print(f"resize_to: {args.resize_to}\n")

    print(f"encoder: {args.encoder}\n")

    print(f"num_slots: {args.num_slots}")
    print(f"slot_att_iter: {args.slot_att_iter}")
    print(f"slot_dim: {args.slot_dim}")
    print(f"query_opt: {args.query_opt}")
    print(f"patch_size: {args.patch_size}")
    print(f"token_num: {args.token_num}")
    print(f"encoder: {args.encoder}")
    print(f"ISA: {args.ISA}\n")

    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"----rank-----: {args.rank}")
    print(f"warmup: {args.warmup}")
    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Dinosaur++")

    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ezvsl_vggss',
                        help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--use_checkpoint', action="store_true")
    
    # === Data Related Parameters ===
    # Data params
    parser.add_argument('--trainset', default='vggss', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)


    # === ViT Related Parameters ===
    parser.add_argument('--encoder', type=str, default="dinov2-vitb-14", 
                        choices=["dinov2-vitb-14", "dino-vitb-16", "dino-vitb-8", "sup-vitb-16"])


    # === Slot Attention Related Parameters ===
    parser.add_argument('--num_slots', type=int, default=4)
    parser.add_argument('--slot_att_iter', type=int, default=3)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--tau', default=0.02, type=float, help='tau')

    parser.add_argument('--query_opt', action="store_true")
    parser.add_argument('--ISA', action="store_true")

    parser.add_argument('--resize_to', nargs='+', type=int, default=[640, 640])

    parser.add_argument("--dropout_img", type=float, default=0.9, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")

    # === Training Related Parameters ===
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup epochs")
    parser.add_argument("--loss1_weight", type=float, default=100, )
    parser.add_argument("--loss2_weight", type=float, default=100, )
    parser.add_argument("--loss3_weight", type=float, default=100, )
    parser.add_argument('--workers', type=int, default=2)


    parser.add_argument('--seed', type=int, default=1234)
    

    args = parser.parse_args()

    set_remaining_args(args)

    return args