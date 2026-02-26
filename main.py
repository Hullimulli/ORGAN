import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from train import train
from config import config_list
import h5py
from datetime import datetime
import random, os, argparse
import ast

def run(args):
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.resume_path is not None:
        save_path = args.resume_path
        config = {}
        with open(os.path.join(save_path, "config.txt"), 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                try:
                    config[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    config[key] = value 

    else:
        save_path = os.path.join(args.savePath,f"{datetime.today().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-4]}_{args.name}_{args.config}_seed{args.seed}")
        os.mkdir(save_path)
        config = config_list[args.config]
        for key, value in vars(args).items():
            if key not in config:
                config[key] = value
        with open(os.path.join(save_path,"config.txt"), 'w') as file:
            for key, value in config.items():
                file.write('%s: %s\n' % (key, value))

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    with h5py.File(config["data_path"], "r") as data:
        data_img = data["img"][:]
        data_lst = data["lst"][:]
        train_size = int(data_img.shape[0] * (1 - config["val_split"]))
        dataset = TensorDataset(
            torch.Tensor(data_img[:train_size]),
            torch.Tensor(data_lst[:train_size, :, :3 + config["list_dim"]])
        )
        dataLoaderTrain = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        dataset = TensorDataset(
            torch.Tensor(data_img[train_size:]),
            torch.Tensor(data_lst[train_size:, :, :3 + config["list_dim"]])
        )
        dataLoaderEval = DataLoader(dataset, batch_size=config["batch_size_val"])

        config["im_dim"] = data_img.shape[1]
        config["num_objects"] = data_lst.shape[1]
        config["im_size"] = data_img.shape[2]
        config["noise_size"] = config["im_size"] // 4
        del data_img, data_lst

    train(dataLoaderTrain, dataLoaderEval, config, save_path, device, args.name, args.wandb_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # Basic Arguments
    parser.add_argument("--name", default=None, help="Experiment Name, leave empty if you don't use wandb")
    parser.add_argument("--resume_path", default=None, help="If given, continues training on the model saved in this path")
    parser.add_argument("--config", default="sprites", help="The index of the config in config.py that has to be loaded")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use")
    parser.add_argument("--numWorkers", type=int, default=0, help="How many workers the dataloader uses")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--savePath", default="./Saves/", help="Save location of the model")
    parser.add_argument("--wandb_path", default="./", help="Save location of wandb log files")
    parser.add_argument("--deterministic", action='store_true', help="Enable deterministic training (slows down training)")
    
    # Loss Weights (Lambdas)
    parser.add_argument("--lambda_cyc_img", type=float, default=100, help="Cycle consistency loss weight for images")
    parser.add_argument("--lambda_cyc_lst", type=float, default=10, help="Cycle consistency loss weight for lists")
    parser.add_argument("--lambda_dis_lst", type=float, default=1.0, help="Discriminator loss weight for lists")
    parser.add_argument("--lambda_dis_img", type=float, default=1.0, help="Discriminator loss weight for images")
    parser.add_argument("--lambda_loc", type=float, default=0.5, help="Location loss weight")
    parser.add_argument("--lambda_pres", type=float, default=15, help="Presence loss weight")
    parser.add_argument("--lambda_entropy", type=float, default=0.0, help="Entropy loss weight")
    parser.add_argument("--lambda_lst_prior", type=float, default=0.0, help="Prior loss weight for lists")
    parser.add_argument("--lambda_im_prior", type=float, default=0.0, help="Prior loss weight for images")
    parser.add_argument("--lambda_gp", type=float, default=0, help="Gradient penalty loss weight")
    
    # Architectures
    parser.add_argument("--g_n_pad", type=int, default=3, help="Number of padding pixels for generators")
    
    ## Image Generator
    parser.add_argument("--im_g_num_channel", type=int, default=64, help="Number of channels in the image generator")
    parser.add_argument("--im_g_num_output_layers", type=int, default=3, help="Number of output layers in the image generator")
    parser.add_argument("--im_g_num_layers", type=int, default=3, help="Number of layers in the image generator")
    
    ## List Generator
    parser.add_argument("--lst_g_num_channel", type=int, default=64, help="Number of channels in the list generator")
    parser.add_argument("--lst_g_num_fc_layers", type=int, default=3, help="Number of fully connected layers in the list generator")
    parser.add_argument("--lst_g_num_conv_layers", type=int, default=4, help="Number of convolutional layers in the list generator")
    parser.add_argument("--lst_g_num_channel_scorer", type=int, default=24, help="Number of channels in the scorer of the list generator")
    parser.add_argument("--hardmax", action='store_true', help="Use hardmax instead of top k")
    
    ## Image Discriminator
    parser.add_argument("--im_dis_num_channels", type=int, default=16, help="Number of channels in the image discriminator")
    parser.add_argument("--im_dis_num_inter_layers", type=int, default=3, help="Number of intermediate layers in the image discriminator")
    
    ## List Discriminator
    parser.add_argument("--lst_dis_transformer_size", type=int, default=16, help="Transformer size in the list discriminator")
    parser.add_argument("--lst_dis_pos_mlp_hidden_dim", type=int, default=32, help="Hidden dimension of position MLP in the list discriminator")
    parser.add_argument("--lst_dis_attn_mlp_hidden_mult", type=int, default=4, help="Multiplier for attention MLP hidden dimension in the list discriminator")
    parser.add_argument("--lst_dis_num_fc_layers", type=int, default=3, help="Number of fully connected layers in the list discriminator")
    parser.add_argument("--lst_dis_num_transformer_layers", type=int, default=3, help="Number of transformer layers in the list discriminator")
    
    # Training Parameters
    parser.add_argument("--memory_size", type=int, default=5, help="Memory size for training")
    parser.add_argument("--transformSpace", action='store_true', help="Enable transformation space usage")
    parser.add_argument("--lr_0_gen", type=float, default=0.0001, help="Learning rate for the generator")
    parser.add_argument("--lr_0_dis", type=float, default=0.0001, help="Learning rate for the discriminator")
    parser.add_argument("--n_warm_up", type=int, default=15, help="Warm-up steps for training of the discriminators")
    parser.add_argument("--loss", default="mae", help="Loss function to use (e.g., 'ssim_yuv', 'mae', 'mse', 'ssim', 'ms_ssim')")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=8, help="Batch size for validation")

    args = parser.parse_args()
    run(args)
