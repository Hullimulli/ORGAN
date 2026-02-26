import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Generator.Im_Gen       import ImGen_ANN
from Generator.List_Gen     import ListGen_ANN
from Discriminator.Im_Dis   import ImDis_ANN
from Discriminator.List_Dis import ListDis_ANN

from eval import run as eval_script
from plot import run as plot_script

import os
from Utilities.losses import FullLoss, SSIM_Loss, MSE_Loss, MAE_Loss, MS_SSIM_Loss, SSIM_YUV_Loss, SSIM_Canny_YUV_Loss
import Utilities.plotting as Plotting
import Utilities.memory   as Memory

os.environ['WANDB_DISABLE_CODE'] = 'True'
os.environ['WANDB_IGNORE_GLOBS'] = '*.patch'
def lr_schedule(epoch, n_epoch, n_epoch_warmup):

    if epoch < n_epoch_warmup:
        return (epoch+1)/n_epoch_warmup
    elif epoch < n_epoch//2:
        return 1
    elif epoch < n_epoch:
        return 2*(1-epoch/n_epoch)
    else:
        return 0

def train(dataloader_train, dataloader_val, config, save_path, device, wandbName: str = None, wandbDirectory: str = None):
    print("Saving in:")
    print(save_path)
    if config["loss"] == "ssim":
        lossfunction = SSIM_Loss(data_range=1,size_average=True,channel=config["im_dim"])
    elif config["loss"] == "mse":
        lossfunction = MSE_Loss()
    elif config["loss"] == "mae":
        lossfunction = MAE_Loss()
    elif config["loss"] == "ms_ssim":
        lossfunction = MS_SSIM_Loss(data_range=1, size_average=True, channel=config["im_dim"], win_size=7)
    elif config["loss"] == "ssim_yuv":
        if config["im_dim"]==3:
            lossfunction = SSIM_YUV_Loss()
        elif config["im_dim"]==1:
             lossfunction = SSIM_Loss(data_range=1,size_average=True,channel=config["im_dim"])
        else:
            raise Exception("SSIM in YUV Space not implemented for the number of image channels provided")
    else:
        raise Exception("Loss function not implemented")
    losses = FullLoss(
        lst_dim=config["list_dim"],
        seg_size=config["seg_size"],
        im_size=config["im_size"],
        lambda_cyc_img=config["lambda_cyc_img"],
        lambda_cyc_lst=config["lambda_cyc_lst"],
        lambda_dis_lst=config["lambda_dis_lst"],
        lambda_dis_img=config["lambda_dis_img"],
        cyc_loss_func=lossfunction,
        lambda_loc=config["lambda_loc"],
        lambda_pres=config["lambda_pres"],
        lambda_entropy=config["lambda_entropy"],
        lambda_gp = config["lambda_gp"],
        lambda_lst_prior = config["lambda_lst_prior"],
        lambda_im_prior = config["lambda_im_prior"],
        n_pad=config["g_n_pad"]
    )

    im_gen   =   ImGen_ANN(data_dim=config["list_dim"],
                           out_dim=config["im_dim"],
                           noise_size=config["noise_size"],
                           im_size=config["im_size"],
                           n_pad=config["g_n_pad"],
                           data_smooth=config["data_smooth"],
                           transformSpace=config["transformSpace"],
                           num_channel=config["im_g_num_channel"],
                           num_output_layers=config["im_g_num_output_layers"],
                           num_layers=config["im_g_num_layers"]).to(device)
    list_gen = ListGen_ANN(sigma=config["sigma"],
                           K=config["num_objects"],
                           data_dim=config["list_dim"],
                           input_size=config["im_size"],
                           seg_size=config["seg_size"],
                           seg_shift=config["seg_shift"],
                           im_dim=config["im_dim"],
                           n_pad=config["g_n_pad"],
                           num_channel_scorer=config["lst_g_num_channel_scorer"],
                           num_channel=config["lst_g_num_channel"],
                           num_fc_layers=config["lst_g_num_fc_layers"],
                           num_conv_layers=config["lst_g_num_conv_layers"]).to(device)
    if config["hardmax"]:
        list_gen.deployed = True
    im_dis   =   ImDis_ANN(im_dim=config["im_dim"],
                           im_size=config["im_size"],
                           num_channels=config["im_dis_num_channels"],
                           num_inter_layers=config["im_dis_num_inter_layers"]).to(device)
    list_dis = ListDis_ANN(data_dim=config["list_dim"],
                        im_size=config["im_size"],
                        transformer_size=config["lst_dis_transformer_size"],
                        pos_mlp_hidden_dim=config["lst_dis_pos_mlp_hidden_dim"],
                        attn_mlp_hidden_mult=config["lst_dis_attn_mlp_hidden_mult"],
                        num_fc_layers=config["lst_dis_num_fc_layers"],
                        num_transformer_layers=config["lst_dis_num_transformer_layers"]
                        ).to(device)
    
    log_file = os.path.join(save_path, "log.txt")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                pass  # Read until the last line like a fool
            if line.startswith("Epoch: "):
                start_epoch = int(line.split(",")[0].split(":")[1].strip())+1
        print(f"Starting from Epoch {start_epoch}")
    else:
        start_epoch = 0
    if wandbName is not None:
        import wandb
        run_id = None
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("wandb_id:"):
                        run_id = line.split("wandb_id:")[-1].strip()
                        break
        if run_id is None:
            run_id = wandb.util.generate_id()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"wandb_id:{run_id}\n")

        config["param_scorer"] = sum(p.numel() for p in list_gen.scorer_ann.parameters())
        config["param_extractor"] = sum(p.numel() for p in list_gen.feature_ann.parameters())
        config["param_img_gen"] = sum(p.numel() for p in im_gen.parameters())
        config["param_list_dist"] = sum(p.numel() for p in list_dis.parameters())
        config["param_img_dist"] = sum(p.numel() for p in im_dis.parameters())
        run = wandb.init(project="ORGAN", dir=wandbDirectory, id=run_id, resume="allow")
        wandb.run.name = wandbName
        config["save_path"] = save_path
        wandb.config.update(config)
    else:
        print(f"#param scorer: {sum(p.numel() for p in list_gen.scorer_ann.parameters())}")
        print(f"#param extractor: {sum(p.numel() for p in list_gen.feature_ann.parameters())}")
        print(f"#param img gen: {sum(p.numel() for p in im_gen.parameters())}")
        print(f"#param list dis: {sum(p.numel() for p in list_dis.parameters())}")
        print(f"#param img dis: {sum(p.numel() for p in im_dis.parameters())}")

    optimizer_gen = optim.Adam(params=list(im_gen.parameters()) + list(list_gen.parameters()), lr=config["lr_0_gen"], betas=(0.5, 0.9))
    optimizer_dis = optim.Adam(params=list(im_dis.parameters()) + list(list_dis.parameters()), lr=config["lr_0_dis"], betas=(0.5, 0.9))

    def update_optimizer_learning_rate_gen(lr):
        for param_group in optimizer_gen.param_groups:
            param_group['lr'] = lr * config["lr_0_gen"]
    def update_optimizer_learning_rate_dis(lr):
        for param_group in optimizer_dis.param_groups:
            param_group['lr'] = lr * config["lr_0_dis"]

    im_real_memory = Memory.Memory(config["memory_size"]*config["batch_size"])
    im_syn_memory = Memory.Memory(config["memory_size"]*config["batch_size"])
    list_real_memory = Memory.Memory(config["memory_size"]*config["batch_size"])
    list_syn_memory = Memory.Memory(config["memory_size"]*config["batch_size"])

    if os.path.exists(os.path.join(save_path, "im_gen.pth")):
        im_gen.load_state_dict(torch.load(os.path.join(save_path, "im_gen.pth"), map_location=device, weights_only=True), strict=False)
        print("Loaded existing image generator")
    else:
        torch.save(im_gen.state_dict(), os.path.join(save_path, "im_gen.pth"))
    if os.path.exists(os.path.join(save_path, "list_gen.pth")):
        list_gen.load_state_dict(torch.load(os.path.join(save_path, "list_gen.pth"), map_location=device, weights_only=True), strict=False)
        print("Loaded existing list generator")
    else:
        torch.save(list_gen.state_dict(), os.path.join(save_path, "list_gen.pth"))
    if os.path.exists(os.path.join(save_path, "im_dis.pth")):
        im_dis.load_state_dict(torch.load(os.path.join(save_path, "im_dis.pth"), map_location=device, weights_only=True), strict=False)
        print("Loaded existing image discriminator")
    else:
        torch.save(im_dis.state_dict(), os.path.join(save_path, "im_dis.pth"))
    if os.path.exists(os.path.join(save_path, "list_dis.pth")):
        list_dis.load_state_dict(torch.load(os.path.join(save_path, "list_dis.pth"), map_location=device, weights_only=True), strict=False)
        print("Loaded existing list discriminator")
    else:
        torch.save(list_dis.state_dict(), os.path.join(save_path, "list_dis.pth"))

    mAverageGen = Memory.movingAverage(25)
    mAverageDis = Memory.movingAverage(25)
    mAverageDisIm = Memory.movingAverage(25)
    mAverageDisLst = Memory.movingAverage(25)
    mAverageGenImCyc = Memory.movingAverage(25)
    mAverageGenLstCyc = Memory.movingAverage(25)
    mAverageGenImReg = Memory.movingAverage(25)
    mAverageGenLstReg = Memory.movingAverage(25)
    mAverageDisImReg = Memory.movingAverage(25)
    mAverageDisLstReg = Memory.movingAverage(25)

    pbar = tqdm(range(start_epoch, config["n_epochs"]), desc="Training")
    for epoch in pbar:
        torch.cuda.empty_cache()
        lr_gen = lr_schedule(epoch, config["n_epochs"], 0)
        lr_dis = lr_schedule(epoch, config["n_epochs"], config["n_warm_up"])
        update_optimizer_learning_rate_gen(lr_gen)
        update_optimizer_learning_rate_dis(lr_dis)
        list_gen.update_lr(lr_gen)
        im_gen.update_lr(lr_gen)
        loss_gen_all = 0
        loss_dis_all = 0
        loss_gen_all_im = 0
        loss_gen_all_lst = 0
        loss_dis_all_im = 0
        loss_dis_all_lst = 0
        loss_dis_all_im_reg = 0
        loss_dis_all_lst_reg = 0
        n_steps = 0
        for step, tensors in enumerate(dataloader_train):
            im_real = tensors[0].to(device)
            list_real = tensors[1].to(device)
            # Train the generators
            im_gen.train()
            list_gen.train()
            im_dis.eval()
            list_dis.eval()

            optimizer_gen.zero_grad()

            list_syn, _ = list_gen.forward(im_real)
            im_syn = im_gen.forward(list_real)
            list_cyc, loc_cyc = list_gen.forward(im_syn)
            im_cyc = im_gen.forward(list_syn)

            res_im_syn = im_dis.forward(im_syn)
            res_list_syn = list_dis.forward(list_syn)
            res_im_real = im_dis.forward(im_real)
            res_list_real = list_dis.forward(list_real)

            loss_gen, loss_gen_cyc_im, loss_gen_cyc_lst = losses.gen_loss(
                res_im_real=res_im_real,
                res_list_real=res_list_real,
                res_im_syn=res_im_syn,
                res_list_syn=res_list_syn,
                im_gt=im_real,
                im_pred=im_cyc,
                list_gt=list_real,
                list_pred=list_cyc,
                patch_pred=loc_cyc,
                seg_loc=list_gen.shifts,
                debug=True
            )

            loss_gen.backward()
            optimizer_gen.step()

            loss_gen = loss_gen.cpu().detach().numpy()
            loss_gen_cyc_im = loss_gen_cyc_im.cpu().detach().numpy()
            loss_gen_cyc_lst = loss_gen_cyc_lst.cpu().detach().numpy()

            # Train the discriminators
            im_gen.eval()
            list_gen.eval()
            im_dis.train()
            list_dis.train()

            optimizer_dis.zero_grad()

            list_syn, _ = list_gen.forward(im_real)
            im_syn = im_gen.forward(list_real)

            im_real_elements = im_real_memory.add_and_return_element(im_real)
            im_syn_elements = im_syn_memory.add_and_return_element(im_syn)
            list_real_elements = list_real_memory.add_and_return_element(list_real)
            list_syn_elements = list_syn_memory.add_and_return_element(list_syn)

            res_im_syn = im_dis.forward(im_syn_elements)
            res_list_syn = list_dis.forward(list_syn_elements)
            res_im_real = im_dis.forward(im_real_elements)
            res_list_real = list_dis.forward(list_real_elements)

            loss_im = losses.im_dis_loss(res_im_real, res_im_syn)
            loss_list = losses.list_dis_loss(res_list_real, res_list_syn)

            # Compute gradient penalty
            # Regularization weight for GP
            if config["lambda_gp"] > 0:
                gp_im = losses.compute_gradient_penalty(im_dis, im_real_elements, im_syn_elements)
                gp_list = losses.compute_gradient_penalty(list_dis, list_real_elements, list_syn_elements)
                loss_dis = (loss_im + loss_list + gp_im + gp_list)
                gp_im = gp_im.cpu().detach().numpy()
                gp_list = gp_list.cpu().detach().numpy()
            else:
                loss_dis = (loss_im + loss_list)
                gp_im = 0
                gp_list = 0

            loss_im = loss_im.cpu().detach().numpy()
            loss_list = loss_list.cpu().detach().numpy()

            loss_dis.backward()
            optimizer_dis.step()

            loss_dis = loss_dis.cpu().detach().numpy()
            if wandbName is not None:
                n_steps += 1
                loss_gen_all += loss_gen
                loss_dis_all += loss_dis
                loss_gen_all_im += loss_gen_cyc_im
                loss_gen_all_lst += loss_gen_cyc_lst
                loss_dis_all_im += loss_im
                loss_dis_all_lst += loss_list
                loss_dis_all_im_reg += gp_im
                loss_dis_all_lst_reg += gp_list
            pbar.set_description(
                f"\rBatch {step}/{len(dataloader_train)}, Epoch: {epoch + 1} | "
                f"L G: {mAverageGen(loss_gen):.3f}, "
                f"L D: {mAverageDis(loss_dis):.3f} | "
                f"L G Im Cyc: {mAverageGenImCyc(loss_gen_cyc_im):.3f}, "
                f"L G Lst Cyc: {mAverageGenLstCyc(loss_gen_cyc_lst):.3f}, "
                f"L D Im: {mAverageDisIm(loss_im):.3f}, "
                f"L D Lst: {mAverageDisLst(loss_list):.3f}, "
                f"L D Im Reg: {mAverageDisImReg(gp_im):.3f}, "
                f"L D Lst Reg: {mAverageDisLstReg(gp_list):.3f} "
                )

        if wandbName is not None:
            wandb.log({
                "Training Loss Generator": loss_gen_all/n_steps,
                "Training Loss Discriminator": loss_dis_all/n_steps,
                "Training Cycle Loss Image Generator": loss_gen_all_im/n_steps,
                "Training Cycle Loss List Generator": loss_gen_all_lst/n_steps,
                "Training Loss Image Discriminator": loss_dis_all_im/n_steps,
                "Training Loss List Discriminator": loss_dis_all_lst/n_steps,
                "Training Regularization Loss Image Discriminator": loss_dis_all_im_reg/n_steps,
                "Training Regularization Loss List Discriminator": loss_dis_all_lst_reg/n_steps,
                 }, 
                step=epoch + 1)

        # Eval the performance
        im_gen.eval()
        list_gen.eval()
        im_dis.eval()
        list_dis.eval()

        losses_gen = []
        losses_im = []
        losses_list = []

        losses_im_syn = []
        losses_im_real = []
        losses_list_syn = []
        losses_list_real = []

        losses_im_cyc = []
        losses_list_cyc = []

        im_real_vis = []
        list_syn_vis = []
        im_cy_vis = []
        list_real_vis = []
        im_syn_vis = []
        list_cyc_vis = []
        im_dis_vis = []
        list_dis_vis = []
        maxVisSamples = 8
        index_VisSamples = 0

        for im_real, list_real in dataloader_val:
            im_real = im_real.to(device)
            list_real = list_real.to(device)

            list_syn, _ = list_gen.forward(im_real)
            im_syn = im_gen.forward(list_real)
            list_cyc, loc_cyc = list_gen.forward(im_syn)
            im_cyc = im_gen.forward(list_syn)

            res_im_syn = im_dis.forward(im_syn)
            res_list_syn = list_dis.forward(list_syn)
            res_im_real = im_dis.forward(im_real)
            res_list_real = list_dis.forward(list_real)

            losses_im_syn.append(torch.mean(res_im_syn).detach().cpu().item())
            losses_list_syn.append(torch.mean(res_list_syn).detach().cpu().item())
            losses_im_real.append(torch.mean(res_im_real).detach().cpu().item())
            losses_list_real.append(torch.mean(res_list_real).detach().cpu().item())

            losses_list_cyc.append(losses.list_cyc_loss(list_gt=list_real,
                                                                   list_pred=list_cyc,
                                                                   patch_pred=loc_cyc,
                                                                   seg_loc=list_gen.shifts).detach().cpu().item())
            losses_im_cyc.append(losses.im_cyc_loss(im_gt=im_real,
                                                               im_pred=im_cyc).detach().cpu().item())

            loss_gen = losses.lambda_cyc_img*losses.im_cyc_loss(im_real,
                                          im_cyc) + \
                       losses.lambda_dis_lst*losses.list_cyc_loss(list_gt=list_real,
                                            list_pred=list_cyc,
                                            patch_pred=loc_cyc,
                                            seg_loc=list_gen.shifts)
            loss_im = losses.im_dis_loss(res_im_real, res_im_syn)
            loss_list = losses.list_dis_loss(res_list_real, res_list_syn)

            if index_VisSamples < maxVisSamples:
                im_real_vis.append(im_real[0].detach().cpu().numpy())
                list_syn_vis.append(list_syn[0].detach().cpu().numpy())
                im_cy_vis.append(im_cyc[0].detach().cpu().numpy())
                list_real_vis.append(list_real[0].detach().cpu().numpy())
                im_syn_vis.append(im_syn[0].detach().cpu().numpy())
                list_cyc_vis.append(list_cyc[0].detach().cpu().numpy())
                im_dis_vis.append(res_im_syn[0].detach().cpu().numpy())
                list_dis_vis.append(res_list_syn[0].detach().cpu().numpy())
                index_VisSamples+=1

            losses_gen.append(loss_gen.detach().cpu().item())
            losses_im.append(loss_im.detach().cpu().item())
            losses_list.append(loss_list.detach().cpu().item())

        data_preds = [np.stack(im_real_vis,axis=0),
                     np.stack(list_syn_vis,axis=0),
                     np.stack(im_cy_vis,axis=0),
                     np.stack(list_real_vis,axis=0),
                     np.stack(im_syn_vis,axis=0),
                     np.stack(list_cyc_vis,axis=0)]
        dist_pred_img = np.stack(im_dis_vis,axis=0)
        dist_pred_lst = np.stack(list_dis_vis, axis=0)

        data_pred_0 = [im_real[0, ...].detach().cpu().numpy(),
                       list_syn[0, ...].detach().cpu().numpy(),
                       im_cyc[0, ...].detach().cpu().numpy(),
                       list_real[0, ...].detach().cpu().numpy(),
                       im_syn[0, ...].detach().cpu().numpy(),
                       list_cyc[0, ...].detach().cpu().numpy()]
        im_stacks = []
        for index in range(maxVisSamples):
            data_pred = [d[index] for d in data_preds]
            im_stacks.append(Plotting.create_plot(*data_pred))
        im_stacks = np.array(im_stacks)
        im_stack_0 = Plotting.create_plot(*data_pred_0)
        coloursImage = np.tile(np.array([1,1,1]).reshape(1, 1, -1), (im_stacks.shape[0], im_stacks.shape[1]//2, 1))
        coloursList = np.tile(np.array([1, 1, 1]).reshape(1, 1, -1), (im_stacks.shape[0], im_stacks.shape[1]//2, 1))
        coloursImage[:, 1] = [0, 0, 0]
        coloursList[:, 1] = [0, 0, 0]
        coloursImage[:, 1, 0] = np.clip(dist_pred_img,0,1)
        coloursList[:, 1, 0] = np.clip(dist_pred_lst,0,1)
        coloursImage[:, 1, 1] = 1-np.clip(dist_pred_img,0,1)
        coloursList[:, 1, 1] = 1-np.clip(dist_pred_lst,0,1)
        stiched_stack_im_cycle = Plotting.stitchImages(im_stacks[:,:3],borderColour=coloursImage)
        stiched_stack_list_cycle = Plotting.stitchImages(im_stacks[:,3:],borderColour=coloursList)

        # Saving the data
        d = {"dis_im_real": np.mean(losses_im_real),
             "dis_im_syn": np.mean(losses_im_syn),
             "dis_list_real": np.mean(losses_list_real),
             "dis_list_syn": np.mean(losses_list_syn),
             "gen_im_cyc": np.mean(losses_im_cyc),
             "gen_list_cyc": np.mean(losses_list_cyc),
             "image_stack_0": im_stack_0,
             "data_pred_0": data_pred_0,
             "data_pred": data_preds
             }

        if wandbName is not None:
            wandb.log({
                "epoch": epoch,
                "Val Loss Img Discriminator Real": d["dis_im_real"],
                "Val Loss Img Discriminator Syn": d["dis_im_syn"],
                "Val Loss List Discriminator Real": d["dis_list_real"],
                "Val Loss List Discriminator Syn": d["dis_list_syn"],
                "Val Loss Img Cycle": d["gen_im_cyc"],
                "Val Loss List Cycle": d["gen_list_cyc"]
            },
            step=epoch + 1)
            plt.imshow(stiched_stack_im_cycle)
            wandb.log({"Image Cycle": wandb.Image(plt)}, step=epoch + 1)
            plt.close()
            plt.imshow(stiched_stack_list_cycle)
            wandb.log({"List Cycle": wandb.Image(plt)}, step=epoch + 1)
            plt.close()
            fig = Plotting.list_to_table(np.stack(list_syn_vis,axis=0)[::2])
            wandb.log({"Predicted List": wandb.Image(fig)}, step=epoch + 1)
            plt.close()

        torch.save(im_gen.state_dict(), os.path.join(save_path,"im_gen.pth"))
        torch.save(list_gen.state_dict(), os.path.join(save_path, "list_gen.pth"))
        torch.save(im_dis.state_dict(), os.path.join(save_path, "im_dis.pth"))
        torch.save(list_dis.state_dict(), os.path.join(save_path, "list_dis.pth"))

        with open(log_file, "a") as f:
            f.write(f"Epoch: {epoch}, " +
                    f"Val Loss Img Discriminator Real: {d['dis_im_real']}, " +
                    f"Val Loss Img Discriminator Syn: {d['dis_im_syn']}, " +
                    f"Val Loss List Discriminator Real: {d['dis_list_real']}, " +
                    f"Val Loss List Discriminator Syn: {d['dis_list_syn']}, " +
                    f"Val Loss Img Cycle: {d['gen_im_cyc']}, " +
                    f"Val Loss List Cycle: {d['gen_list_cyc']}\n")

    class Args:
        gpus = device.index
        numWorkers = 0
        k = config["k_test"]
        checkDis = True
        path = save_path
        batch_size = 1
        dataPath = config["data_path_test"]
        sureness = None
        run_plot_script = True
        no_ground = config["no_ground"]
    args = Args()
    ssim, mae = eval_script(args)
    print(ssim, mae)
    print(args.no_ground)
    countDiff, mean_dist, _, _, _ = plot_script(args)
    print(countDiff, mean_dist)
    if wandbName is not None and not args.no_ground:
        wandb.log({
            "Count Difference": countDiff,
            "Location Error": mean_dist,
            "SSIM": ssim,
            "MAE": mae
        }, step=config["n_epochs"])