import random, os, argparse
import numpy as np
from tqdm import tqdm
from Utilities.plotting import stitchImages, markObjects, list_to_table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.metrics   import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
import h5py
import ast

def run(args):
    config = {}
    imagesToShowSideLength = 5
    max_distance = 10
    imagesToShow = imagesToShowSideLength**2
    if os.path.isfile(os.path.join(args.path, "config.txt")):
        config = {}
        with open(os.path.join(args.path, "config.txt"), 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                try:
                    config[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    config[key] = value 
    else:
        # Used for data produced by other models than ORGAN
        config = {
            "seg_size": 28
        }
    with h5py.File(args.dataPath, "r") as h5_file:
        data_images = np.transpose(h5_file["img"][:],axes=(0,2,3,1))
        data_lists = h5_file["lst"][:]

    if not os.path.isfile(args.path):
        path_file = os.path.join(args.path, "test_results.h5")
        path = args.path
    else:
        path_file = args.path
        path = os.path.dirname(args.path)

    seg_size  = int(config["seg_size"])
    with h5py.File(path_file, "r") as h5_file:
        predictions = h5_file["lst"][:] if "lst" in h5_file else None
        dis_list_predictions = h5_file["dis_lst_result"][:] if "dis_lst_result" in h5_file else None
        features = h5_file["fts"][:] if "fts" in h5_file else None
        images = h5_file["im_cyc"][:] if "im_cyc" in h5_file else None
    images = np.transpose(images,axes=(0,2,3,1))
    cycle_img = stitchImages(np.transpose(np.stack((data_images[:4], images[:4])), axes=(1, 0, 2, 3, 4)))
    marked_images = np.reshape(markObjects(data_images[:imagesToShow],predictions[:imagesToShow],obj_size=seg_size),(imagesToShowSideLength, imagesToShowSideLength) + data_images.shape[1:3] + (3,))

    if features is not None and np.size(features)>1:
        features = np.transpose(features, axes=(0, 2, 3, 1))
        if len(features) > 16:
            features = features.reshape(16,16,features.shape[1],features.shape[2],features.shape[3])
            illustrations_features = stitchImages(np.transpose(features, axes=(1, 0, 2, 3, 4)))
        else:
            illustrations_features = stitchImages(features[np.newaxis])
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_bad(color='white')
    if dis_list_predictions is not None:
        colours = cmap(dis_list_predictions[:imagesToShow])[:,:3]
        illustrations = stitchImages(marked_images, borderColour = np.reshape(colours,(imagesToShowSideLength, imagesToShowSideLength, 3)))
    else:
        illustrations = stitchImages(marked_images)
    if not args.no_ground:
        distances = []
        y_differences = []
        x_differences = []
        n_obj_detected_true = np.zeros(len(predictions))
        n_obj_detected_false = np.zeros(len(predictions))
        n_obj_ground = np.zeros(len(data_lists))
        for i in range(len(predictions)):
            existing_objects = np.argwhere(data_lists[i, :, 2] > 0.5)[:,0]
            X_t = data_lists[i, existing_objects, 0]
            Y_t = data_lists[i, existing_objects, 1]
            indices_pred = np.argwhere(predictions[i, :, 2] > 0.5)[:, 0]
            X_p = predictions[i, indices_pred, 0]
            Y_p = predictions[i, indices_pred, 1]
            A = (X_t[:, np.newaxis] - X_p[np.newaxis, :]) ** 2 + (Y_t[:, np.newaxis] - Y_p[np.newaxis, :]) ** 2
            l1, l2 = linear_sum_assignment(A)
            pred_sort = predictions[i,indices_pred][l2]
            data_lst_sort = data_lists[i,existing_objects][l1]
            x_difference = pred_sort[:, 0] - data_lst_sort[:, 0]
            y_difference = pred_sort[:, 1] - data_lst_sort[:, 1]
            distance = np.sqrt(x_difference ** 2 + y_difference ** 2)
            n_obj_detected_true[i] = np.count_nonzero(distance <= max_distance, axis=-1)
            n_obj_detected_false[i] = np.count_nonzero(distance > max_distance, axis=-1) + len(indices_pred)-len(pred_sort)
            n_obj_ground[i] = len(existing_objects)
            distances.append(distance)
            y_differences.append(y_difference)
            x_differences.append(x_difference)

        distances = np.concatenate(distances)
        x_differences = np.concatenate(x_differences)
        y_differences = np.concatenate(y_differences)

    distributions_existence, edges_existence = np.histogram(
        predictions.flatten(),
        bins=1000,
        range=(0,
               1),
        density=True
    )
    edges_existence = edges_existence[:-1]
    distributions_existence = distributions_existence / np.sum(distributions_existence)

    if predictions.shape[-1] > 3:
        distributions_feat = np.zeros((predictions.shape[-1]-3,1000))
        edges_feat = np.zeros((predictions.shape[-1]-3,1001))
        minVal = np.min(predictions[:, :, 3:])
        maxVal = np.max(predictions[:, :, 3:])
        for i in range(len(distributions_feat)):
            distributions_feat[i], edges_feat[i] = np.histogram(
                predictions[predictions[...,2]>0.5, i+3].flatten(),
                bins=1000,
                range=(minVal,
                       maxVal),
                density=True
            )
        distributions_feat = distributions_feat / np.sum(distributions_feat, axis=1, keepdims=True)
        if not args.no_ground:
            distributions_feat_ground = np.zeros((data_lists.shape[-1]-3,1000))
            edges_feat_ground = np.zeros((data_lists.shape[-1]-3,1001))
            minVal = np.min(data_lists[:, :, 3:])
            maxVal = np.max(data_lists[:, :, 3:])
            for i in range(len(distributions_feat_ground)):
                distributions_feat_ground[i], edges_feat_ground[i] = np.histogram(
                    data_lists[data_lists[...,2]>0.5, i+3].flatten(),
                    bins=1000,
                    range=(minVal,
                           maxVal),
                    density=True
                )
            edges_feat_ground = edges_feat_ground[..., :-1]
            distributions_feat_ground = distributions_feat_ground / np.sum(distributions_feat_ground, axis=1, keepdims=True)
        edges_feat = edges_feat[...,:-1]
    pdf = PdfPages(os.path.join(path, "results.pdf"))

    plt.figure(dpi=300)
    plt.imshow(illustrations)
    plt.axis('off')
    pdf.savefig()
    plt.close()

    plt.figure(dpi=300)
    plt.plot(edges_existence, distributions_existence, label=f'Existence Prob.', linewidth=3)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('p(x)', fontsize=16)
    plt.title('Existence Probability Distribution', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    pdf.savefig()
    plt.close()

    if predictions.shape[-1] > 3:
        plt.figure(dpi=300)
        for i in range(len(distributions_feat)):
            plt.plot(edges_feat[i], distributions_feat[i], label=f'Feature {i}', linewidth=3)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('p(x)', fontsize=16)
        plt.title('Predicted Feature distribution', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        pdf.savefig()
        plt.close()

        if not args.no_ground:
            plt.figure(dpi=300)
            for i in range(len(distributions_feat_ground)):
                plt.plot(edges_feat_ground[i], distributions_feat_ground[i], label=f'Feature {i}', linewidth=3)
            plt.xlabel('x', fontsize=16)
            plt.ylabel('p(x)', fontsize=16)
            plt.title('Ground Feature distribution', fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            pdf.savefig()
            plt.close()

    plt.figure(dpi=300)
    plt.imshow(cycle_img)
    plt.axis('off')
    pdf.savefig()
    plt.close()

    if features is not None and np.size(features)>1:
        plt.figure(dpi=300)
        plt.imshow(illustrations_features)
        plt.axis('off')
        pdf.savefig()
        plt.close()
    countDiff = None
    mean_dist = None
    f1s = None
    precisions = None
    recalls = None
    if not args.no_ground:
        cm = confusion_matrix(n_obj_ground, n_obj_detected_true+n_obj_detected_false)
        color1 = np.array((1, 1, 0))
        color2 = np.array((1, 0, 0))
        colors = [(1, 1, 1)]
        for i in range(np.max(cm)):
            colors.append(tuple(color1 + (color2 - color1) * i / (np.max(cm) - 1)))
        colormap = LinearSegmentedColormap.from_list("white_yellow_red", colors)
        plt.figure(figsize=(15, 12))
        countDiff = np.mean(np.abs(n_obj_ground - (n_obj_detected_true+n_obj_detected_false)))
        plt.title(f"Avg. count difference: {countDiff:.2f}")
        plt.imshow(cm, cmap=colormap)
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    plt.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=30)
        pdf.savefig()
        plt.close()


        # Calculate mean and standard deviation of distances
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        plt.figure(figsize=(10, 10))
        plt.title(f'Differences: Mean = {mean_dist:.2f}, Std = {std_dist:.2f}')
        plt.plot(x_differences, y_differences, 'k.', alpha=0.25)
        plt.xlim([-14, 14])
        plt.ylim([-14, 14])
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Unique values in the ground truth (possible object counts)
        unique_counts = np.unique(n_obj_ground)
        unique_counts = unique_counts[unique_counts>0]
        # Calculate accuracy for each possible value in the ground truth array
        precisions = []
        recalls = []
        f1s = []
        for count in unique_counts:
            mask = n_obj_ground == count
            precision = np.where((n_obj_detected_true[mask]+n_obj_detected_false[mask]) == 0, 0, n_obj_detected_true[mask] / (n_obj_detected_true[mask]+n_obj_detected_false[mask]))
            precision[precision==np.nan]=1
            recall = n_obj_detected_true[mask] / count # Accuracy for this count
            f1 = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
            precisions.append(np.mean(precision))
            recalls.append(np.mean(recall))
            f1s.append(np.mean(f1))
        f1s = np.array(f1s)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        # Plot the accuracy
        plt.plot(unique_counts, precisions, marker='o', linestyle='-', label="Precision")
        plt.plot(unique_counts, recalls, marker='o', linestyle='-', label="Recall")
        plt.plot(unique_counts, f1s, marker='o', linestyle='-', label="F1")
        plt.xlabel('#Objects')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

    pdf.close()

    return countDiff, mean_dist, f1s, precisions, recalls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--path",
                        help="save location")
    parser.add_argument("--dataPath",
                        help="save location")
    #parser.add_argument("--sureness", default=None)
    parser.add_argument("--no_ground", action="store_true",
                        help="Disable ground truth")
    args = parser.parse_args()
    run(args)