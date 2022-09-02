from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
sys.path.append('./')
from datasets.generated import load_data as load_generated_data
from datasets.rwth import load_data
from utils.loops import train, test
from utils.misc import train_val_dataset

def prepare_evaluation():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train_mode', type=str, default="real", help="training mode for using real and generated data, can be any of ['real', 'gen', 'reggen', 'mixgen', 'wgen'].")
    parser.add_argument("--dset1", type=str, default="none", help="specify the directory of the folder that contains real images.")
    parser.add_argument("--dset2", type=str, default="none", help="specify the directory of the folder that contains generated images.")
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size for evaluation")
    parser.add_argument("--dims", default=64, type=int, help="image dimensions sizes")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
    parser.add_argument("-save", "--save_dir", type=str, default="./")

    parser.add_argument("--seed", type=int, default=-1, help="seed for generating random numbers")
    parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl", help="cuda backend for DDP training \in ['nccl', 'gloo']")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    world_size = gpus_per_node * args.total_nodes

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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
    train_dataset = Dataset_(data_name='rwth-train',
                           data_dir=args.dset1,
                           train=True,
                           crop_long_edge=True,
                           resize_size=[args.dim, args.dim],
                           random_flip=True,
                           normalize=True,
                           hdf5_path=None,
                           load_data_in_memory=True,
                           pose=False)
    test_dataset = Dataset_(data_name='rwth-test',
                           data_dir=args.dset1,
                           train=False,
                           crop_long_edge=True,
                           resize_size=[args.dim, args.dim],
                           random_flip=True,
                           normalize=True,
                           hdf5_path=None,
                           load_data_in_memory=True,
                           pose=False)
    gen_dataset = Dataset_(data_name='rwth-gen-spade',
                           data_dir=args.dset2,
                           train=True,
                           crop_long_edge=True,
                           resize_size=[args.dim, args.dim],
                           random_flip=True,
                           normalize=True,
                           hdf5_path=None,
                           load_data_in_memory=True,
                           pose=False)
    train_dataset, val_dataset = train_val_dataset(dataset = train_dataset, random_state = args.seed)
    if local_rank == 0:
        print("Size of train dataset: {dataset_size}".format(dataset_size=len(train_dataset)))
        print("Size of validation dataset: {dataset_size}".format(dataset_size=len(val_dataset)))
        print("Size of test dataset: {dataset_size}".format(dataset_size=len(test_dataset)))
        print("Size of generated dataset: {dataset_size}".format(dataset_size=len(gen_dataset)))

    # -----------------------------------------------------------------------------
    # define dataloaders for real and generated datasets.
    # -----------------------------------------------------------------------------
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True) 
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True) 
    gen_dataloader = DataLoader(dataset=gen_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=None,
                                drop_last=True,
                                persistent_workers=True)
    dataloaders = {'train': train_dataloader,
                    'val': val_dataloader,
                    'test': test_dataloader,
                    'gen': gen_dataloader}

    # -----------------------------------------------------------------------------
    # load a network (Efficientnet v2).
    # -----------------------------------------------------------------------------
    model = models.efficientnet_v2_m(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_dataset.classes)
    model = model.to(device)

    # -----------------------------------------------------------------------------
    # train the model.
    # -----------------------------------------------------------------------------
    model, train_results = train(model, dataloaders, epochs = args.epochs, mode = train_mode, device = local_rank)
    if not args.save_dir is None:
        with open(args.save_dir+'train_results{}.pkl'.format(i), 'wb') as f:
            pickle.dump(train_results, f)

    # -----------------------------------------------------------------------------
    # test the model.
    # -----------------------------------------------------------------------------
    test_results = test(model, dataloaders, device = local_rank)
    if not args.save_dir is None:
        with open(args.save_dir+'test_results.pkl', 'wb') as f:
            pickle.dump(test_results, f)

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
