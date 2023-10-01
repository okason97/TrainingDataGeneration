from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from torch.backends import cudnn
import torch
import random
import numpy as np
import os
import pickle
import sys
sys.path.append('./')
from utils.loops import train, test, filter, create_confusion_matrix, calculate_class_accuracy
import utils.misc as misc
from data_util import Dataset_
from pathlib import Path

def prepare_evaluation():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train_mode', type=str, default="real", help="training mode for using real and generated data, can be any of ['real', 'gen', 'reggen', 'mixgen', 'wgen'].")
    parser.add_argument("--dset1", type=str, default="none", help="specify the directory of the folder that contains real images.")
    parser.add_argument("--dset2", type=str, default="none", help="specify the directory of the folder that contains generated images.")
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size for evaluation")
    parser.add_argument("--dims", default=64, type=int, help="image dimensions sizes")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument("-sm", "--save_model", action="store_true")
    parser.add_argument("-lm", "--load_model", type=str, default=None)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("-cm", action="store_true", help="confusion matrix")
    parser.add_argument("--class_accuracy", action="store_true", help="calculate class accuracy")
    parser.add_argument("--n_classes", default=13, type=int, help="number of classes")
    parser.add_argument("--dset_used", default=1., type=float, help="percentage of the dataset to be used when training the classifier")
    parser.add_argument("--val_size", default=0.25, type=float, help="validation size")
    parser.add_argument("--growth", action="store_true", help="regularization growth or decay.")
    parser.add_argument("--reg_alpha", default=1., type=float, help="regularization loss starting weight.")
    parser.add_argument("--reg_beta", default=-1., type=float, help="regularization loss weight change.")
    parser.add_argument("-l", "--load_data_in_memory", action="store_true", help="put the whole train dataset on the main memory for fast I/O")
    parser.add_argument("--imagenet_weights", action="store_true")

    parser.add_argument("--seed", type=int, default=-1, help="seed for generating random numbers")
    parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl", help="cuda backend for DDP training \in ['nccl', 'gloo']")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    world_size = gpus_per_node * args.total_nodes

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.seed == -1: args.seed = random.randint(1, 4096)
    if world_size == 1: print("You have chosen a specific GPU. This will completely disable data parallelism.")
    return args, world_size, gpus_per_node, rank

def classify(local_rank, args, world_size, gpus_per_node):
    # -----------------------------------------------------------------------------
    # determine cuda, cudnn, and backends settings.
    # -----------------------------------------------------------------------------
    cudnn.benchmark, cudnn.deterministic = False, True

    # -----------------------------------------------------------------------------
    # initialize all processes and fix seed of each process
    # -----------------------------------------------------------------------------
    if args.distributed_data_parallel:
        global_rank = args.current_node * (gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, world_size, args.backend)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    misc.fix_seed(args.seed + global_rank)

    # -----------------------------------------------------------------------------
    # load real and generated datasets.
    # -----------------------------------------------------------------------------
    train_dataset = Dataset_(data_name='train',
                           data_dir=args.dset1,
                           train=True,
                           crop_long_edge=True,
                           resize_size=[args.dims, args.dims],
                           random_flip=True,
                           normalize=True,
                           hdf5_path=None,
                           load_data_in_memory=args.load_data_in_memory,
                           pose=False)
    test_dataset = Dataset_(data_name='test',
                           data_dir=args.dset1,
                           train=False,
                           crop_long_edge=True,
                           resize_size=[args.dims, args.dims],
                           random_flip=True,
                           normalize=True,
                           hdf5_path=None,
                           load_data_in_memory=args.load_data_in_memory,
                           pose=False)
    if args.dset2 != 'none':
        gen_dataset = Dataset_(data_name='gen',
                               data_dir=args.dset2,
                               train=True,
                               crop_long_edge=True,
                               resize_size=[args.dims, args.dims],
                               random_flip=True,
                               normalize=True,
                               hdf5_path=None,
                               load_data_in_memory=args.load_data_in_memory,
                               pose=False)
    if args.dset_used>1:
        dset_used = int(args.dset_used)
    elif args.dset_used != 1:
        dset_used = 1-args.dset_used
    else:
        dset_used = args.dset_used
    train_dataset, val_dataset = misc.train_val_dataset(dataset = train_dataset, val_split=args.val_size, random_state = args.seed)
    if dset_used != 1:
        train_dataset, _ = misc.train_val_dataset(dataset = train_dataset, val_split=None, train_size=dset_used, random_state = args.seed)
    if local_rank == 0:
        print("Size of train dataset: {dataset_size}".format(dataset_size=len(train_dataset)))
        print("Size of validation dataset: {dataset_size}".format(dataset_size=len(val_dataset)))
        print("Size of test dataset: {dataset_size}".format(dataset_size=len(test_dataset)))
        if args.dset2 != 'none':
            print("Size of generated dataset: {dataset_size}".format(dataset_size=len(gen_dataset)))

    # -----------------------------------------------------------------------------
    # define dataloaders for real and generated datasets.
    # -----------------------------------------------------------------------------
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True) 
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True) 
    if args.dset2 != 'none':
        gen_dataloader = DataLoader(dataset=gen_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    sampler=None,
                                    drop_last=True,
                                    persistent_workers=True)
    else:
        gen_dataloader = None
    dataloaders = {'train': train_dataloader,
                    'val': val_dataloader,
                    'test': test_dataloader,
                    'gen': gen_dataloader}

    # -----------------------------------------------------------------------------
    # load a network (Efficientnet v2).
    # -----------------------------------------------------------------------------
    if args.imagenet_weights:
        print("Using imagenet weights.")
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_m(weights=weights)
    else:
        print("Not using imagenet weights.")
        model = models.efficientnet_v2_m()
    num_ftrs = model.classifier[1].in_features
    model.fc = nn.Linear(num_ftrs, args.n_classes)
    model = model.to(local_rank)

    # -----------------------------------------------------------------------------
    # train the model.
    # -----------------------------------------------------------------------------
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
    else:
        model, train_results = train(model, dataloaders, epochs = args.epochs, growth=args.growth, reg_alpha=args.reg_alpha, reg_beta=args.reg_beta, mode = args.train_mode, device = local_rank)
        if not args.save_dir is None:
            with open(args.save_dir+'train_results.pkl', 'wb') as f:
                pickle.dump(train_results, f)
        if args.save_model:
            torch.save(model.state_dict(), args.save_dir+'model_state_dict.pt')

    # -----------------------------------------------------------------------------
    # test the model.
    # -----------------------------------------------------------------------------
    test_results = test(model, dataloaders, device = local_rank)
    if not args.save_dir is None:
        with open(args.save_dir+'test_results.pkl', 'wb') as f:
            pickle.dump(test_results, f)

    # -----------------------------------------------------------------------------
    # filter the top K generated samples.
    # -----------------------------------------------------------------------------
    if args.top_k>0:
        if args.dset2 != 'none':
            gen_dataset = misc.dataset_with_indices(Dataset_)(data_name='gen',
                                data_dir=args.dset2,
                                train=True,
                                crop_long_edge=True,
                                resize_size=[args.dims, args.dims],
                                random_flip=False,
                                normalize=False,
                                hdf5_path=None,
                                load_data_in_memory=args.load_data_in_memory,
                                pose=False)

            gen_dataloader = DataLoader(dataset=gen_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=args.num_workers,
                                        sampler=None,
                                        drop_last=True,
                                        persistent_workers=True)
            dataloaders = {'gen': gen_dataloader}
            filter(model, dataloaders, top_k = args.top_k, n_classes = args.n_classes, save_dir = os.path.join(args.save_dir,os.path.basename(os.path.normpath(args.dset2)),'train'), device = local_rank)
        else:
            print('No dataset to filter.')

    # -----------------------------------------------------------------------------
    # filter the top K generated samples.
    # -----------------------------------------------------------------------------
    if args.cm:
        create_confusion_matrix(model, dataloaders, save_dir = args.save_dir, device = local_rank)

    # -----------------------------------------------------------------------------
    # calculate class accuracy.
    # -----------------------------------------------------------------------------
    if args.class_accuracy:
        calculate_class_accuracy(model, dataloaders, save_dir = args.save_dir, device = local_rank)

if __name__ == "__main__":
    args, world_size, gpus_per_node, rank = prepare_evaluation()

    if args.distributed_data_parallel and world_size > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        try:
            torch.multiprocessing.spawn(fn=classify,
                                        args=(args,
                                              world_size,
                                              gpus_per_node),
                                        nprocs=gpus_per_node)
        except KeyboardInterrupt:
            misc.cleanup()
    else:
        classify(local_rank=rank,
                 args=args,
                 world_size=world_size,
                 gpus_per_node=gpus_per_node)
