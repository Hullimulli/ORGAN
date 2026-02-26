import random, os, argparse
import numpy as np
import Utilities.datasets as Datasets
from torchvision import transforms
from torchvision.datasets import MNIST
from Utilities.plotting import stitchImages, list2im_
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import h5py

def write_dataset(path: str, img: np.ndarray, lst: np.ndarray, labels: np.ndarray = None):
    with h5py.File(path + ".h5", "w") as h5_file:
        h5_file.create_dataset("img", data=img, compression="gzip", compression_opts=3)
        h5_file.create_dataset("lst", data=lst, compression="gzip", compression_opts=3)
        if labels is not None:
            h5_file.create_dataset("labels", data=labels)
    if len(img)>=64:
        n_img = 8
    else:
        n_img = 2
    img = np.transpose(img[:n_img ** 2], axes=(0, 2, 3, 1))
    lst_img = np.stack([list2im_(l, img.shape[1:3], point_size=7) for l in lst[:n_img ** 2]])
    images = stitchImages(np.reshape(img, (n_img, n_img, img.shape[1], img.shape[2], img.shape[3])))
    lists = stitchImages(np.reshape(lst_img, (n_img, n_img, img.shape[1], img.shape[2], 3)))
    pdf = PdfPages(path + ".pdf")
    plt.figure(dpi=300)
    plt.imshow(images)
    plt.axis('off')
    pdf.savefig()
    plt.close()
    plt.figure(dpi=300)
    plt.imshow(lists)
    plt.axis('off')
    pdf.savefig()
    plt.close()
    pdf.close()

def create_dataset(args):
    if args.dataset == "sprites":
        settings = {
            "num_elements": 10,
            "im_size": 128,
            "object_size": 28,
            "min_distance": -10,
            "alpha": 0.5,
            "data_dim": 3,
            "step_size": 1
                    }
        img, lst = Datasets.create_dataset_sprites(10000, seed=42, settings=settings)
        write_dataset(args.path, img, lst)
    elif args.dataset == "sprites_test":
        settings = {
            "num_elements": 10,
            "im_size": 128,
            "object_size": 28,
            "min_distance": -10,
            "alpha": -1,
            "data_dim": 3,
            "step_size": 1
                    }
        img, lst = Datasets.create_dataset_sprites(10000, seed=66, settings=settings, test_set=True)
        write_dataset(args.path, img, lst)
    elif args.dataset == "sprites_big":
        settings = {
            "num_elements": 40,
            "im_size": 256,
            "object_size": 28,
            "min_distance": -10,
            "alpha": 0.5,
            "data_dim": 3,
            "step_size": 1
                    }
        img, lst = Datasets.create_dataset_sprites(10000, seed=42, settings=settings, test_set=False)
        write_dataset(args.path, img, lst)
    elif args.dataset == "sprites_big_few_features":
        settings = {
            "num_elements": 15,
            "im_size": 256,
            "object_size": 28,
            "min_distance": 20,
            "alpha": 1,
            "data_dim": 3,
            "step_size": 1
                    }
        img, lst = Datasets.create_dataset_sprites(135, seed=66, settings=settings, test_set=True, fixed_features=3)
        write_dataset(args.path, img, lst)
    elif args.dataset == "sprites_big_test":
        settings = {
            "num_elements": 40,
            "im_size": 256,
            "object_size": 28,
            "min_distance": -10,
            "alpha": -1,
            "data_dim": 3,
            "step_size": 1
                    }
        img, lst = Datasets.create_dataset_sprites(10000, seed=66, settings=settings, test_set=True)
        write_dataset(args.path, img, lst)
    elif args.dataset == "sprites_test_very_big":
        settings = {
            "num_elements": 125,
            "im_size": 512,
            "object_size": 28,
            "min_distance": -10,
            "alpha": -1,
            "data_dim": 3,
            "step_size": 10
                    }
        img, lst = Datasets.create_dataset_sprites(2500, seed=66, settings=settings, test_set=True)
        write_dataset(args.path, img, lst)
    elif args.dataset == "mnist":
        settings = {
            "num_elements": 10,
            "im_size": 128,
            "object_size": 28,
            "min_distance": -15,
            "alpha": -1,
            "data_dim": 32,
            "step_size": 1
                    }
        mnist = MNIST(root='./Data', train=True, download=True, transform=transforms.ToTensor())
        data = mnist.data.numpy()
        targets = mnist.targets.numpy()
        img, lst, labels, img_test, lst_test, labels_test = Datasets.create_dataset_mnist(10000, seed=42, mnist=data, targets=targets, settings=settings)
        write_dataset(args.path, img, lst, labels)
        write_dataset(args.path + "_test", img_test, lst_test, labels_test)
    elif args.dataset == "cells":
        num_elem = 10
        settings = {
            "num_elements": num_elem,
            "im_size": 128,
            "object_size": 5,
            "min_distance": 20,
            "alpha": 0.5,#0.685,
            "data_dim": 2,
            "step_size": 1
                    }
        img, lst, img_test, lst_test = Datasets.create_dataset_cells(10000, seed=42, settings=settings)
        write_dataset(args.path, img, lst)
        write_dataset(args.path+"_test", img_test, lst_test)
    elif args.dataset == "cells_big":
        num_elem = 40
        settings = {
            "num_elements": num_elem,
            "im_size": 256,
            "object_size": 5,
            "min_distance": 20,
            "alpha": 0.5,
            "data_dim": 2,
            "step_size": 1
                    }
        img, lst, img_test, lst_test = Datasets.create_dataset_cells(64, seed=42, settings=settings)
        write_dataset(args.path, img, lst)
        write_dataset(args.path + "_test", img_test, lst_test)
    elif args.dataset == "cells_large":
        num_elem = 160
        settings = {
            "num_elements": num_elem,
            "im_size": 512,
            "object_size": 5,
            "min_distance": 20,
            "alpha": 0.5,
            "data_dim": 2,
            "step_size": 1
                    }
        img, lst, img_test, lst_test = Datasets.create_dataset_cells(64, seed=42, settings=settings)
        write_dataset(args.path, img, lst)
        write_dataset(args.path + "_test", img_test, lst_test)
    elif args.dataset == "cells_very_large":
        num_elem = 360
        settings = {
            "num_elements": num_elem,
            "im_size": 768,
            "object_size": 5,
            "min_distance": 20,
            "alpha": 0.5,
            "data_dim": 2,
            "step_size": 1
                    }
        img, lst, img_test, lst_test = Datasets.create_dataset_cells(64, seed=42, settings=settings)
        write_dataset(args.path, img, lst)
        write_dataset(args.path + "_test", img_test, lst_test)
    elif args.dataset == "tetris":
        settings = {
            "object_size": 20,
            "data_dim": 32,
        }
        img, lst, img_test, lst_test = Datasets.create_dataset_tetris(100000, seed=42, settings=settings)
        write_dataset(args.path, img, lst)
        write_dataset(args.path + "_test", img_test, lst_test)
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not implemented.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", default="sprites", help="Name of the data type.")
    parser.add_argument("--path", default="./Data/sprites", help="Filepath where the dataset is saved.")
    args = parser.parse_args()
    create_dataset(args)
 