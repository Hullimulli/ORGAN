import numpy as np
import matplotlib.pyplot as plt
import hashlib
import cv2
def list2im_(l,im_size,point_size=5):
    if isinstance(im_size,int):
        im_size = (im_size,im_size)
    im = np.zeros((im_size[0],im_size[1],3))
    dims = l.shape[1]-3

    for i in range(l.shape[0]):
        x0 = max(int(l[i,0]+0.5)-point_size//2,0)
        x1 = min(int(l[i,0]+0.5)+point_size-point_size//2,im_size[0]-1)
        y0 = max(int(l[i,1]+0.5)-point_size//2,0)
        y1 = min(int(l[i,1]+0.5)+point_size-point_size//2,im_size[1]-1)

        if dims < 3:
            im[x0:x1,y0:y1,:dims+1] = np.clip(l[i,2:]*l[i,2:3],0,1)
        else:
            # Create an array of indices
            indices = np.arange(l[i, 3:].size)
            # Combine the array and indices into a 2D array
            combined = np.vstack((l[i, 3:], indices))
            # Compute a hash of the combined array
            hash_bytes = hashlib.sha256(combined.tobytes()).digest()
            # Convert the first 3 bytes of the hash to an RGB color
            rgb_color = np.array([1 - 0.95 * (b / 255) for b in hash_bytes[:dims]])
            im[x0:x1, y0:y1] = np.clip(rgb_color * l[i, 2:3], 0, 1)[:3]

    return im

def list_to_table(l):
    # Create a figure and a subplot
    fig, axs = plt.subplots(l.shape[0], 1, figsize=(10, 10), facecolor='none')
    # Define the column labels
    col_labels = ['x', 'y', r'$\alpha$'] + [fr'$\eta_{{{i + 1}}}$' for i in range(l.shape[2] - 3)]

    if l.shape[0]==1:
        ax = axs
        ax.axis('tight')
        ax.axis('off')
        # Get the data for the current subplot
        table_data = np.round(l[0],2)
        # Create a table and add it to the plot
        table = ax.table(cellText=table_data,
                 colLabels=col_labels,
                 cellLoc='center',
                 loc='center')
        table.scale(0.3, 1.5)
    else:

        for i in range(l.shape[0]):
            ax = axs[i]
            ax.axis('tight')
            ax.axis('off')
            # Get the data for the current subplot
            table_data = np.round(l[i],2)
            # Create a table and add it to the plot
            table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')
            table.scale(0.3, 1.5)

    return fig

def create_plot(im_real,list_syn,im_cyc,list_real,im_syn,list_cyc):
    factor  = 3//im_real.shape[0]
    im_size = im_real.shape[1]

    im_real = np.tile(np.transpose(im_real,[1,2,0]),[1,1,factor])
    im_syn  = np.tile(np.transpose(im_syn ,[1,2,0]),[1,1,factor])
    im_cyc  = np.tile(np.transpose(im_cyc ,[1,2,0]),[1,1,factor])

    list_real = list2im_(list_real,im_size)
    list_syn  = list2im_(list_syn ,im_size)
    list_cyc  = list2im_(list_cyc ,im_size)

    im_stack  = np.stack([im_real,list_syn,im_cyc,list_real,im_syn,list_cyc],0)

    return im_stack

def create_plot_simple(im_real_1,list_syn_1,im_cyc_1,im_real_2,list_syn_2,im_cyc_2):
    factor  = 3//im_real_1.shape[0]
    im_size = im_real_1.shape[1]

    im_real_1 = np.tile(np.transpose(im_real_1,[1,2,0]),[1,1,factor])
    im_cyc_1  = np.tile(np.transpose(im_cyc_1 ,[1,2,0]),[1,1,factor])

    list_syn_1  = list2im_(list_syn_1 ,im_size)

    im_real_2 = np.tile(np.transpose(im_real_2,[1,2,0]),[1,1,factor])
    im_cyc_2  = np.tile(np.transpose(im_cyc_2 ,[1,2,0]),[1,1,factor])

    list_syn_2  = list2im_(list_syn_2 ,im_size)

    im_stack  = np.stack([im_real_1,list_syn_1,im_cyc_1,im_real_2,list_syn_2,im_cyc_2],0)

    return im_stack

def stitchImages(im_stack, borderColour: np.ndarray = np.array([1,1,1]), border_size: int = 6, colour_size: int = 2):
    if np.ndim(borderColour) == 1:
        borderColour = np.tile(borderColour.reshape(1, 1, -1), (im_stack.shape[0], im_stack.shape[1], 1))
    image = np.ones(((len(im_stack)+1)*border_size+len(im_stack)*im_stack[0].shape[1],(im_stack[0].shape[0]+1)*border_size+im_stack[0].shape[0]*im_stack[0].shape[2],3))
    for m,stage in enumerate(im_stack):
        for n,i in enumerate(stage):
            image[
            m * (border_size + i.shape[0])+border_size-colour_size:m * (border_size + i.shape[0])+border_size+colour_size + i.shape[0],
            n * (border_size + i.shape[1])+border_size-colour_size:n * (border_size + i.shape[1])+border_size+colour_size + i.shape[1]
            ] = borderColour[m,n]

            image[
            m*(border_size+i.shape[0])+border_size:m*(border_size+i.shape[0])+border_size+i.shape[0],
            n*(border_size+i.shape[1])+border_size:n*(border_size+i.shape[1])+border_size+i.shape[1]
            ] = i

    return np.clip(image, 0, 1)

def markObjects(im_stack, lst_stack, border_size: int = 2, obj_size: int = 28):
    markedStack = np.copy(im_stack)
    if markedStack.shape[-1] == 1:
        markedStack = np.concatenate((markedStack,) * 3, axis=-1)
    dims = lst_stack.shape[2] - 3
    for j,l in enumerate(lst_stack):
        for i in range(l.shape[0]):
            im = np.zeros(markedStack[j].shape[:2])
            x0 = max(int(l[i, 0] + 0.5) - obj_size // 2, 0)
            x1 = min(int(l[i, 0] + 0.5) + obj_size - obj_size // 2, im.shape[0] - 1)
            y0 = max(int(l[i, 1] + 0.5) - obj_size // 2, 0)
            y1 = min(int(l[i, 1] + 0.5) + obj_size - obj_size // 2, im.shape[1] - 1)

            if dims < 3:
                colour = np.clip((l[i, 3:]/2+0.5), 0, 1)
                im[x0:x1, y0:y1] = 1
                im[x0+border_size:x1-border_size, y0+border_size:y1-border_size] = 0
            else:
                # Create an array of indices
                indices = np.arange(l[i, 3:].size)
                # Combine the array and indices into a 2D array
                combined = np.vstack((l[i, 3:], indices))
                # Compute a hash of the combined array
                hash_bytes = hashlib.sha256(combined.tobytes()).digest()
                # Convert the first 3 bytes of the hash to an RGB color
                rgb_color = np.array([1 - 0.95 * (b / 255) for b in hash_bytes[:3]])
                colour = np.clip(rgb_color, 0, 1)[:3]
                im[x0:x1, y0:y1] = 1
                im[x0 + border_size:x1 - border_size, y0 + border_size:y1 - border_size] = 0

            markedStack[j][im!=0,:dims] = (1-l[i, 2:3])*markedStack[j][im!=0,:dims] + l[i, 2:3]*colour
    return np.clip(markedStack,0,1)
def plot(im_stack):
    im_stack = np.clip(im_stack,0,1)

    plt.figure(figsize=(15,10))
    for i in range(6):
        x = i//3
        y = i %3
        plt.subplot(2,3,1+i)
        plt.imshow(im_stack[i,:,:,:])
        plt.axis("off")
    plt.show()