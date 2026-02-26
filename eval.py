import random, os, argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from Generator.List_Gen import ListGen_ANN
from Generator.Im_Gen import ImGen_ANN
from Discriminator.Im_Dis import ImDis_ANN
from Discriminator.List_Dis import ListDis_ANN
from tqdm import tqdm
from Utilities.losses import SSIM_YUV_Loss, MAE_Loss
from Utilities.memory import movingAverage
from torch.nn import DataParallel
import h5py
from plot import run as plot_script
import time
import ast

def run(args):
    if os.path.isdir(args.path):
        results_dir = os.path.join(args.path, "test_results")
    else:
        results_dir = args.path
    if not isinstance(args.gpus,list):
        args.gpus = [args.gpus]
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    config = {}

    save_path = args.path
    config = {}
    with open(os.path.join(save_path, "config.txt"), 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            try:
                config[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                config[key] = value 

    with h5py.File(args.dataPath, "r") as h5_file:
        data = h5_file["img"][:]
        lists = h5_file["lst"][:, :, :3 + int(config["list_dim"])]
    dataset = TensorDataset(torch.Tensor(data), torch.Tensor(lists))
    ssim_average = movingAverage(len(data))
    mae_average = movingAverage(len(data))
    ssim_loss = SSIM_YUV_Loss(lambda_mae=0)
    mae_loss = MAE_Loss()
    config["im_dim"] = data.shape[1]
    config["im_size"] = (data.shape[2],data.shape[3])
    config["noise_size"] = (config["im_size"][0] // 4,config["im_size"][1] // 4)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    list_gen = ListGen_ANN(sigma=config["sigma"],
                           K=args.k,
                           data_dim=config["list_dim"],
                           input_size=(data.shape[2],data.shape[3]),
                           seg_size=config["seg_size"],
                           seg_shift=config["seg_shift"],
                           im_dim=config["im_dim"],
                           n_pad=config["g_n_pad"],
                           num_channel_scorer=config["lst_g_num_channel_scorer"],
                           num_channel=config["lst_g_num_channel"],
                           num_fc_layers=config["lst_g_num_fc_layers"],
                           num_conv_layers=config["lst_g_num_conv_layers"]).to(device)
    im_gen = ImGen_ANN(data_dim=config["list_dim"],
                       out_dim=config["im_dim"],
                       noise_size=config["noise_size"],
                       im_size=config["im_size"],
                       n_pad=config["g_n_pad"],
                       data_smooth=config["data_smooth"],
                       transformSpace=config["transformSpace"],
                       num_channel=config["im_g_num_channel"],
                       num_output_layers=config["im_g_num_output_layers"],
                       num_layers=config["im_g_num_layers"]).to(device)
    if args.checkDis:
        config["im_size"] = max(config["im_size"])
        im_dis   =   ImDis_ANN( 
            im_dim=config["im_dim"],
            im_size=config["im_size"],
            num_channels=config["im_dis_num_channels"],
            num_inter_layers=config["im_dis_num_inter_layers"]
            ).to(device)
        list_dis = ListDis_ANN(data_dim=config["list_dim"],
                        im_size=config["im_size"],
                        transformer_size=config["lst_dis_transformer_size"],
                        pos_mlp_hidden_dim=config["lst_dis_pos_mlp_hidden_dim"],
                        attn_mlp_hidden_mult=config["lst_dis_attn_mlp_hidden_mult"],
                        num_fc_layers=config["lst_dis_num_fc_layers"],
                        num_transformer_layers=config["lst_dis_num_transformer_layers"]
                        ).to(device)
    del data
    list_gen.load_state_dict(torch.load(os.path.join(args.path, "list_gen.pth"), map_location=device, weights_only=True), strict=False)
    list_gen.deployed = True
    list_gen.scorer_ann.deployed = True
    list_gen.eval()
    im_gen.load_state_dict(torch.load(os.path.join(args.path, "im_gen.pth"), map_location=device, weights_only=True), strict=False)
    im_gen.eval()
    if args.checkDis:
        im_dis.load_state_dict(torch.load(os.path.join(args.path, "im_dis.pth"), map_location=device, weights_only=True), strict=False)
        im_dis.eval()
        list_dis.load_state_dict(torch.load(os.path.join(args.path, "list_dis.pth"), map_location=device, weights_only=True), strict=False)
        list_dis.eval()
    predicted_list = np.zeros((len(dataLoader)*args.batch_size, args.k, 3+int(config["list_dim"])))
    i = 0
    im_cyc = []
    lst_cyc = []
    dis_list_results = np.zeros((len(dataLoader)*args.batch_size)) + np.nan
    dis_img_result = []
    n_saved_img = 16

    tic = time.time()
    for step, tensors in enumerate(dataLoader):
        with torch.no_grad():
            im_real = tensors[0].to(device)
            list_syn, _ = list_gen.forward(im_real)
    print(f"Inference time for {len(dataLoader)*args.batch_size} images: {time.time()-tic}s")

    progress_bar = tqdm(total=len(dataLoader), desc="Predicting")
    for step, tensors in enumerate(dataLoader):
        progress_bar.update(1)
        with torch.no_grad():
            im_real = tensors[0].to(device)
            list_syn, _ = list_gen.forward(im_real)
            lst_real = tensors[1].to(device)
            gen_list, _ = list_gen.forward(im_gen.forward(lst_real))
            gen_img = im_gen.forward(list_syn)
            mae_average(mae_loss(im_real,gen_img,factor=1).detach().cpu().numpy())
            ssim_average(ssim_loss(im_real,gen_img).detach().cpu().numpy())
            if args.checkDis:
                dis_lst_result = list_dis.forward(list_syn)
                if args.checkDis:
                    dis_img_result.append(im_dis.forward(gen_img).detach().cpu().numpy())
                i += len(im_real)
            if i < n_saved_img:
                lst_cyc.append(gen_list.detach().cpu().numpy())
                im_cyc.append(gen_img.detach().cpu().numpy())
        predicted_list[step*args.batch_size:step*args.batch_size+list_syn.size(dim=0)] = list_syn.detach().cpu().numpy()
        if args.checkDis:
            dis_list_results[step * args.batch_size:step * args.batch_size + dis_lst_result.size(dim=0)] = dis_lst_result.detach().cpu().numpy()
    im_cyc = np.concatenate(im_cyc,axis=0)[:n_saved_img]
    lst_cyc = np.concatenate(lst_cyc,axis=0)[:n_saved_img]
    if args.checkDis:
        dis_img_result = np.concatenate(dis_img_result,axis=0)[:n_saved_img]
        print("Dist Gen:",np.nanmean(dis_list_results))
    del list_gen

    if int(config["list_dim"]) > 0 and int(config["list_dim"]) < 3:
        list_mapping = np.zeros((16 ** int(config["list_dim"]), args.k, 3+int(config["list_dim"])))
        list_mapping[:, 0, 0] = int(config["seg_size"])//2
        list_mapping[:, 0, 1] = int(config["seg_size"])//2
        list_mapping[:, 0, 2] = 0.99
        if int(int(config["list_dim"])) > 1:
            for i in range(16):
                list_mapping[i::16, 0, 3] = np.arange(16) / 15
                list_mapping[i * 16:(i + 1) * 16, 0, 4] = np.arange(16) / 15
        else:
            list_mapping[:, 0, 3] = np.arange(16) / 15
        list_mapping = torch.from_numpy(list_mapping).to(device).float()
        dataset = TensorDataset(list_mapping)
        dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        predicted_features = np.zeros((len(dataLoader)*args.batch_size, int(config["im_dim"]),int(config["seg_size"]),int(config["seg_size"])))
        progress_bar = tqdm(total=len(dataLoader), desc="Illustrating")
        for step, tensors in enumerate(dataLoader):
            progress_bar.update(1)
            with torch.no_grad():
                feature_list = tensors[0].to(device)
                im_syn = im_gen.forward(feature_list)
            predicted_features[step*args.batch_size:step*args.batch_size+args.batch_size] = im_syn.detach().cpu().numpy()[:,:,:int(config["seg_size"]),:int(config["seg_size"])]
    else:
        predicted_features = np.empty(0)
    print(f"Saving to {results_dir}.h5")
    with h5py.File(results_dir + ".h5", "w") as h5_file:
        h5_file.create_dataset("im_cyc", data=im_cyc)
        h5_file.create_dataset("lst_cyc", data=lst_cyc)
        if args.checkDis:
            h5_file.create_dataset("dis_img_result", data=dis_img_result)
            h5_file.create_dataset("dis_lst_result", data=dis_list_results)
        h5_file.create_dataset("fts", data=predicted_features)
        h5_file.create_dataset("lst", data=predicted_list)
        h5_file.create_dataset("ssim", data=np.float32(1-ssim_average()))
        h5_file.create_dataset("mae", data=np.float32(mae_average()))

    if not args.no_ground:
        countDiff, mean_dist, f1s, precisions, recalls = plot_script(args)
        with h5py.File(results_dir + ".h5", "a") as h5_file:
            h5_file.create_dataset("count_diff", data=np.float32(countDiff))
            h5_file.create_dataset("mean_dis", data=np.float32(mean_dist))
            h5_file.create_dataset("f1", data=f1s)
            h5_file.create_dataset("precision", data=precisions)
            h5_file.create_dataset("recall", data=recalls)
    return 1-ssim_average(), mae_average()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gpus", default='0', nargs="+",
                        help="used GPU")
    parser.add_argument("--numWorkers", default=0,
                        help="number of workers for the dataloader")
    parser.add_argument("--k", help="number of patches evaluated")
    parser.add_argument("--checkDis", action="store_true")
    parser.add_argument("--no_ground", action="store_true")
    parser.add_argument("--path", help="save location")
    parser.add_argument("--batch_size", default=1,
                        help="Size of evaluation batch")
    parser.add_argument("--dataPath", help="save location of data")
    args = parser.parse_args()
    args.k = int(args.k)
    run(args)
