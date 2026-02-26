import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import rotate, zoom
import cv2
import os
from typing import List
import tensorflow.compat.v1 as tf
from skimage import exposure
import itertools

# Disable eager execution for compatibility with TensorFlow 1.x code
tf.disable_eager_execution()

def sample_from_histogram(hist, uni_samples):
    bin_edges = np.linspace(0,1,len(hist))
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    indices = np.searchsorted(cdf, uni_samples)
    samples = bin_edges[indices]

    return samples

def create_list(
        list_size: int,
        num_elements: int,
        im_size: int,
        object_size: int,
        min_distance: float,
        alpha: float,
        data_dim: int,
        data_dist: List[np.ndarray] = None,
        seed: int=0,
        stepSize: int=1,
        fixed_features: int = -1
) -> np.ndarray:
    """
    :param list_size: Number of unique lists (i.e. batch size)
    :param num_elements: Number of elements per list
    :param im_size: Size of the total image
    :param object_size: the size of each object (important to describe the boundary and overlap between objects)
    :param min_distance: Minimal distance (L_infty norm) of two objects. Can be negative. Then, overlaps are allowed
    :param alpha: Probability that each object is actually really there (must be in [0,1])
    :param data_dim: Dimensionality of the list
    :param seed: Seed
    :return: Generated List
    """
    
    np.random.seed(seed)

    lst        = np.random.random((list_size,num_elements,3+data_dim))
    if alpha < 0:
        lst[:, :, 2] = 0.01
        s = lst.shape[0] // (num_elements//stepSize)
        for i,c in enumerate(range(num_elements,0,-stepSize)):
            lst[(i-1)*s:i*s, :c, 2] = 0.98
        np.random.shuffle(lst)

        # Shuffle the second dimension (columns) by generating a random permutation of column indices
        lst = lst[:, np.random.permutation(lst.shape[1])]
    else:
        lst[:,:,2] = (lst[:,:,2]<alpha)*0.98 + 0.01

    if data_dist is not None:
        for i, d in enumerate(data_dist):
            lst[:, :, 3+i] = sample_from_histogram(d, lst[:, :, 3+i])
    if fixed_features!=-1:
        # Create the fixed values (here evenly spaced between 0.2 and 0.9)
        values = np.linspace(0.2, 0.9, fixed_features)
        # Generate all possible combinations for the extra features
        combos = list(itertools.product(values.tolist(), repeat=data_dim))
        combos = np.array(combos)  # shape: (fixed_features**n_extra, n_extra)
        total_combos = combos.shape[0]
        for b in range(list_size):
            # Assign the extra features for batch 'b'
            lst[b, :, 3:] = combos[b % total_combos]

    i = 0
    while i < list_size:
        count    = 0
        elements = 0
        mask     = np.zeros((im_size,im_size))
        while count < 1000 and elements < num_elements:
            pos_x = np.random.random()*(im_size-object_size)+object_size/2
            pos_y = np.random.random()*(im_size-object_size)+object_size/2
            
            # If point is not allowed, try again
            if mask[int(pos_x),int(pos_y)] == 1:
                count += 1
                continue
            
            # The point is allowed, once we are here
            
            # Cut out the region from mask that is not allowed
            x0 = max(0,min(im_size-1,int(pos_x-object_size-min_distance)))
            x1 = max(0,min(im_size-1,int(pos_x+object_size+min_distance)))
            y0 = max(0,min(im_size-1,int(pos_y-object_size-min_distance)))
            y1 = max(0,min(im_size-1,int(pos_y+object_size+min_distance)))
            
            mask[x0:x1,y0:y1] = 1
            
            # Set the points into the list:
            lst[i,elements,0] = pos_x
            lst[i,elements,1] = pos_y
            elements         += 1
        
        if elements == num_elements:
            # This means we finished this particular list; go to the next
            i += 1
        else:
            # This means we ran into issues with that list. Redo it
            print(i,"redo")
            pass
    
    return lst

def create_dataset_sprites(dataset_size,seed,settings,sigma=0,test_set=False,fixed_features=-1):
    num_elements = settings["num_elements"]
    im_size      = settings["im_size"]
    object_size  = settings["object_size"]
    min_distance = settings["min_distance"]
    alpha        = settings["alpha"]
    data_dim     = settings["data_dim"]
    stepSize     = settings["step_size"]
    # Simple dataset
    lst_im = create_list(list_size=dataset_size,
                         num_elements=num_elements,
                         im_size=im_size,
                         object_size=object_size,
                         min_distance=min_distance,
                         alpha=alpha,
                         data_dim=data_dim,
                         seed=seed,
                         stepSize=stepSize,
                         fixed_features=fixed_features)
    lst = create_list(list_size=dataset_size,
                      num_elements=num_elements,
                      im_size=im_size,
                      object_size=object_size,
                      min_distance=min_distance,
                      alpha=alpha,
                      data_dim=data_dim,
                      seed=seed+1,
                      stepSize=stepSize,
                      fixed_features=fixed_features)
    
    max_value = 0.95
    min_value = 0.0

    def generate_decay_field(image_size, centers_y, centers_x, shapes, scales, obj_size=28, peak_value=1.0):
        """
        Generates a decay field where each shape is stored in a separate channel using full NumPy vectorization.

        Parameters:
            image_size (tuple): Size of the image (height, width).
            centers_x (array-like): Array of x-coordinates for multiple centers.
            centers_y (array-like): Array of y-coordinates for multiple centers.
            shapes (array-like): Array of shape factors (0 = circular, 1 = square).
            scales (array-like): Array of scale factors controlling thresholds.
            obj_size (int): Size of each object in pixels.
            peak_value (float): Maximum intensity at each center.

        Returns:
            np.ndarray: Multi-channel decay field (shape: [height, width, num_objects]).
        """
        height, width = image_size
        num_objects = len(centers_x)

        # Create 2D coordinate grids
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Expand to (height, width, num_objects) for vectorized computation
        y = np.expand_dims(y, axis=-1)  # Shape: (height, width, 1)
        x = np.expand_dims(x, axis=-1)  # Shape: (height, width, 1)

        # Reshape centers to match (1, 1, num_objects)
        centers_x = np.array(centers_x).reshape(1, 1, num_objects)
        centers_y = np.array(centers_y).reshape(1, 1, num_objects)

        # Compute absolute distances
        dy = np.abs(y - centers_y)  # Shape: (height, width, num_objects)
        dx = np.abs(x - centers_x)

        # Compute shape-dependent norm transformation (vectorized)
        shapes = np.array(shapes).reshape(1, 1, num_objects)
        norm = np.clip(shapes**2 * 4.5 + 0.5, 0.5, 5.0)

        # Compute generalized distance (vectorized for all objects)
        distance = (dx**norm + dy**norm) ** (1 / norm)

        # Compute decay field using exponential decay
        decay_field = peak_value * np.exp(-2.5 / obj_size * distance)

        # Apply thresholds for each shape (vectorized)
        scales = np.array(scales).reshape(1, 1, num_objects)
        thresholds = scales * 0.3 + 0.35
        binary_mask = decay_field > thresholds  # Shape: (height, width, num_objects)

        return binary_mask.astype(float)

    img = np.zeros((dataset_size,3,im_size,im_size)) + min_value
    img[:,2,:,:] = max_value
    for i in tqdm(range(dataset_size)):
        im = generate_decay_field((im_size,im_size), lst_im[i,:,0], lst_im[i,:,1], lst_im[i,:,4], lst_im[i,:,5], object_size).transpose((2,0,1))
        im = (lst_im[i,:,2]>0.5)[:,np.newaxis,np.newaxis]*im
        img[i,0,:,:] += np.sum(im * lst_im[i,:,3,np.newaxis,np.newaxis],0)*max_value
        img[i,1,:,:] += np.sum(im * (1-lst_im[i,:,3,np.newaxis,np.newaxis]),0)*max_value
        img[i,2,:,:] -= np.sum(im * 1,0)*(max_value-min_value)
    img = np.clip(img,0,1)
    if sigma != 0:
        img += np.random.normal(loc=0.0,scale=sigma/255,size=img.shape)
        img = np.clip(img,0,1)
    if test_set:
        return img,lst_im
    return img,lst

def create_dataset_mnist(dataset_size,seed,mnist,targets,settings,sigma=0,test_set=False):
    # mnist should have size N x 28 x 28
    
    num_elements = settings["num_elements"]
    im_size      = settings["im_size"]
    object_size  = settings["object_size"]
    min_distance = settings["min_distance"]
    alpha        = settings["alpha"]
    data_dim = settings["data_dim"]

    lst_im = create_list(list_size=dataset_size*2,
                         num_elements=num_elements,
                         im_size=im_size,
                         object_size=object_size,
                         min_distance=min_distance,
                         alpha=alpha,
                         data_dim=data_dim,
                         seed=seed)
    lst = create_list(list_size=dataset_size*2,
                      num_elements=num_elements,
                      im_size=im_size,
                      object_size=object_size,
                      min_distance=min_distance,
                      alpha=alpha,
                      data_dim=data_dim,
                      seed=seed+1)

    max_value = 1.0
    min_value = 0.0
    img = np.zeros((dataset_size*2,1,im_size,im_size)) + min_value
    labels = np.zeros((dataset_size*2,num_elements,2))

    indices   = np.random.permutation(mnist.shape[0])
    indices_  = []
    indices__ = []
    for i in range(num_elements):
        indices_.append( indices[i::num_elements])
        indices__.append(indices[i::num_elements])
        
    shortest_length = np.min([indices__[i].shape[0] for i in range(num_elements)])
    j = 0
    while shortest_length < dataset_size*2:
        j += 1
        for i in range(num_elements):
            to_add = indices_[(i+j)%num_elements]
            to_add = to_add[np.random.permutation(to_add.shape[0])]
            indices__[i] = np.concatenate([indices__[i],to_add],0)
        shortest_length = np.min([indices__[i].shape[0] for i in range(num_elements)])
        
    for i in range(dataset_size*2):
        for j in range(num_elements):
            element = lst_im[i,j,:]
            if element[2] < 0.5:
                continue
            try:
                img[i,:,int(element[0]-14):int(element[0]+14),int(element[1]-14):int(element[1]+14)] += mnist[indices__[j][i],np.newaxis,:,:]*max_value
                labels[i,j, 0] = 1
                labels[i,j, 1] = targets[indices__[j][i]]
            except:
                print(i,j,element)
                print(indices__[j][i])
                print(element[0]-14,element[0]+14,element[1]-14,element[1]+14)
                print(mnist[indices__[j][i],np.newaxis,:,:].shape)
                print(img[i,:,element[0]-14:element[0]+14,element[1]-14:element[1]+14].shape)
    
    img = np.clip(img,0,1)
    if sigma != 0:
        img += np.random.normal(loc=0.0,scale=sigma/255,size=img.shape)
        img = np.clip(img,0,1)
    return img[:dataset_size],lst[:dataset_size], labels[:dataset_size], img[dataset_size:],lst_im[dataset_size:], labels[dataset_size:]

def create_dataset_cells(dataset_size,seed,settings):

    num_elements = settings["num_elements"]
    im_size      = settings["im_size"]
    object_size  = settings["object_size"]
    min_distance = settings["min_distance"]
    alpha        = settings["alpha"]
    data_dim     = settings["data_dim"]

    lst = create_list(list_size=dataset_size+2500,
                      num_elements=num_elements,
                      im_size=im_size,
                      object_size=object_size,
                      min_distance=min_distance,
                      alpha=alpha,
                      data_dim=data_dim,
                      seed=seed + 1)

    X = np.tile(np.arange(im_size)[:, np.newaxis], [1, im_size])
    Y = X.T

    def is_green(pixel):
        return np.sum(np.abs(pixel - color)) < 0.3

    filenames = []
    ims = []
    fileNameDir = './Data/Cells/Data/'
    # Load the images
    for filename in os.listdir(fileNameDir):
        if filename.endswith('.JPG') and not filename.startswith('._'):
            filenames.append(filename)
    color = np.array([0.64, 0.70, 0.80])
    color_ = color[np.newaxis, np.newaxis, :]
    progress_bar = tqdm(total=len(filenames), desc="Processing", unit="item")
    if len(filenames)==0:
        raise Exception("No pictures found")
    for filename in filenames:
        progress_bar.update(1)
        n = 1
        data_large = np.array(Image.open(fileNameDir + filename)) / (255. * n * n)
        offset0 = (data_large.shape[0] - (data_large.shape[0] // n) * n) // 2
        offset1 = (data_large.shape[1] - (data_large.shape[1] // n) * n) // 2
        data_large = data_large[offset0:offset0 + (data_large.shape[0] // n) * n,
                     offset1:offset1 + (data_large.shape[1] // n) * n,
                     :]
        data = np.zeros_like(data_large[::n, ::n, :])
        for i in range(n):
            for j in range(n):
                data += data_large[i::n, j::n, :]

        ims.append(data)

    progress_bar.close()

    img = []

    stride = im_size//2
    for image in ims:
        # Calculate the number of patches along the height and width
        num_patches_h = (image.shape[0] - im_size) // stride + 1
        num_patches_w = (image.shape[1] - im_size) // stride + 1

        # Define the shape of the patches and strides
        shape = (num_patches_h, num_patches_w, im_size, im_size, 3)
        strides = (
            stride * image.strides[0], 
            stride * image.strides[1], 
            image.strides[0], 
            image.strides[1],
            image.strides[2] 
        )
        # Use as_strided to create a view with the new shape and strides
        patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
        # Reshape to have patches as individual elements
        img.append(patches.reshape(-1, im_size, im_size, 3))
    img_train = img[:next(i for i, x in enumerate(img) if sum(arr.shape[0] for arr in img[:i+1]) >= dataset_size)]
    img_test = img[len(img_train):]
    img_train = np.concatenate(img_train, 0)
    img_test = np.concatenate(img_test, 0)
    indices = np.random.permutation(np.arange(len(img_train)))
    img_train = img_train[indices]
    indices = np.random.permutation(np.arange(len(img_test)))
    img_test = img_test[indices]
    return img_train.transpose((0,3,1,2)), lst[:len(img_train)], img_test.transpose((0,3,1,2)), lst[len(lst)-len(img_test):]


def create_dataset_tetris(dataset_size,seed,settings, padding=0):

    num_elements = 3
    im_size      = [35,35]
    object_size  = settings["object_size"]
    min_distance = None
    alpha        = 0.99
    data_dim     = settings["data_dim"]

    # Function to decode the TFRecords dataset
    COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
    IMAGE_SIZE = [35, 35]
    MAX_NUM_ENTITIES = 4
    BYTE_FEATURES = ['mask', 'image']
    
    features = {
        'image': tf.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
        'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES] + IMAGE_SIZE + [1], tf.string),
        'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        'color': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
        'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    }
    
    def _decode(example_proto):
        single_example = tf.parse_single_example(example_proto, features)
        for k in BYTE_FEATURES:
            single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8), axis=-1)
        return single_example

    raw_dataset = tf.data.TFRecordDataset(
        "./Data/tetrominoes_train.tfrecords", compression_type=COMPRESSION_TYPE, buffer_size=None
    )
    # Create dataset from TFRecords
    raw_dataset =  raw_dataset.map(_decode, num_parallel_calls=None)
    
    # Prepare datasets for each feature
    image_dataset = []
    mask_dataset = []
    x_dataset = []
    y_dataset = []
    shape_dataset = []
    color_dataset = []
    visibility_dataset = []
    # Iterate through the TFRecord dataset
    n_samples = 0
    progress_bar = tqdm(total=2*dataset_size, desc="Processing", unit="item")
    with tf.Session() as sess:
        iterator = raw_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        try:
            while 2*dataset_size!=n_samples:
                example = sess.run(next_element)
                # Append data to respective lists
                image_dataset.append(example['image'])
                mask_dataset.append(example['mask'])
                x_dataset.append(example['x'])
                y_dataset.append(example['y'])
                shape_dataset.append(example['shape'])
                color_dataset.append(example['color'])
                visibility_dataset.append(example['visibility'])
                n_samples+=1
                progress_bar.update(1)
        except tf.errors.OutOfRangeError:
            pass
        
    offset = np.array([
        [0,0],
        [2,9.5],
        [9.5,2],
        [4.5,7],
        [4.5,7],
        [4.5,7],
        [4.5,7],
        [7,4.5],
        [7,4.5],
        [7,4.5],
        [7,4.5],
        [4.5,7],
        [4.5,7],
        [4.5,4.5],
        [7,4.5],
        [4.5,7],
        [4.5,7],
        [7,4.5],
        [7,4.5],
        [4.5,4.5],
    ])
    shape_dataset = np.array(shape_dataset)
    y_loc = np.array(y_dataset)+offset[shape_dataset.astype(int),1] + padding
    x_loc = np.array(x_dataset)+offset[shape_dataset.astype(int),0] + padding
    img = np.array(image_dataset).astype(np.float32).transpose(0,3,1,2) / 255
    img = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    loc = np.concatenate((x_loc[:,1:,None],y_loc[:,1:,None]),axis=2)
    lst_im = np.zeros((len(img),num_elements,3+data_dim))
    lst_im[...,:2] = loc
    lst_im[...,2] = 0.99
    lst_im[...,3:] = np.random.random((lst_im.shape[0],lst_im.shape[1],data_dim))

    img_permutation = np.random.permutation(lst_im.shape[0])
    elem_permutation = np.random.permutation(lst_im.shape[1])

    color_dataset = np.array(color_dataset)
    lst_im[...,4] = (np.sum(color_dataset[:,1:]*np.array([[1,2,4]]),axis=-1) - 1)  / 6
    lst_im[...,3] = shape_dataset[:,1:] / 20

    # Permute the array
    lst = lst_im[img_permutation, :, :]
    lst = lst[:, elem_permutation, :]

    return img[:dataset_size], lst[:dataset_size], img[dataset_size:dataset_size+dataset_size], lst_im[dataset_size:dataset_size+dataset_size]