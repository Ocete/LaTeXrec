from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf
import linecache

'''
Functions for loading the datasets relevant to the experiments.
'''

def get_paths(dataset):
    """
    Returns the list of important paths for the chosen dataset.

    Params:
    - dataset: if 0 -> toy_dataset; if 1 -> im2latex dataset.
    """

    if dataset == 0:
        base_path = Path('/storage/toy_dataset/')
        images_dir = base_path / 'toy_images_50k'
        filename2line_train = base_path / 'im2latex_toy_50k.lst'
        formulas_file = base_path / 'toy_50k_formulas.lst'
    elif dataset == 1:
        base_path = Path('/storage/dataset_tmp/')
        images_dir = base_path / 'formula_images'
        filename2line_train = base_path / 'im2latex.lst'
        formulas_file = base_path / 'im2latex_formulas.lst'
    else:
        print('Dataset not found')
        return []

    return [base_path, images_dir, filename2line_train, formulas_file]

def load_toy_dataset():
    base_path, images_dir, filename2line_train, formulas_file = get_paths(0)
    return load_dataset(base_path, images_dir, filename2line_train, formulas_file)

def load_im2latex_dataset():
    base_path, images_dir, filename2line_train, formulas_file = get_paths(1)
    return load_dataset(base_path, images_dir,
                        filename2line_train, formulas_file,
                        im2latex_configuration=True)

def load_dataset(base_path,
                 images_dir,
                 filename2line,
                 formulas_file,
                 im2latex_configuration=False):
    """
    Load a dataset.

    Params:
    - base_path: directory in which it is contained.
    - images_dir: directory containing the images.
    - filename2line: path to file containing the filename -> line in formulas
      file mapping.
    - formulas_file: path to file containing the list of formulas.
    - im2latex_configuration: whether to use the configuration for the im2latex
      datasets.
    """

    # If possible, load a previously generated DataFrame containing the list of
    # image file names. Else, generate it.
    if Path(base_path / 'image_files.json').exists():
        image_files = pd.read_json(base_path / 'image_files.json')
    else:
        image_files = [f.relative_to(images_dir)
                       for f in images_dir.rglob('*.png')]
        image_files = pd.DataFrame(list(image_files), columns=['filename'])
        image_files.to_json(base_path / 'image_files.json',
                            default_handler=str)

    # Generate a DataFrame containing the list of image file names along with
    #their corresponding LaTeX formulas.
    if im2latex_configuration:
        fn2l_df = pd.read_csv(filename2line, sep=' ', names=[
                              'line', 'filename', '_'])
        fn2l_df = fn2l_df.drop(['_'], axis=1)
    else:
        fn2l_df = pd.read_csv(filename2line,
                              sep=' ', names=['line', 'filename'])

    # Fix dtype of column being 'object'
    image_files.filename = image_files.filename.astype(str)
    if im2latex_configuration:
        fn2l_df.filename = fn2l_df.filename.astype(
            str).map(lambda x: x+'.png')
    else:
        fn2l_df.filename = fn2l_df.filename.astype(str)
    fn2l_df = pd.merge(image_files, fn2l_df, on='filename')

    # Add formulas to DataFrame
    def get_formula(index):
        f = linecache.getline(str(formulas_file), int(index)+1).strip()
        return f
    linecache.checkcache()
    fn2l_df['formula'] = fn2l_df['line'].apply(get_formula)

    return image_files, fn2l_df

def read_imgs_from_fn2l(fn2l_df,
                        images_dir,
                        from_i,
                        to_i,
                        resize=True):

    """
    Load a certain range of images [from_i:to_i] from the given dataframe and
    returns new dataframe with them.

    Params:
    - fn2l_df: dataframe where the images paths are obtained.
    - images_dir: directory containing the images.
    - from_i: index of the first image to be loaded.
    - to_i: index of where to stop reading images.
    - resize: whether all the images should be resized to the same height.
    """

    new_df = fn2l_df[from_i:to_i].copy(deep=True)

    # Read the images and resize them to the same height, preserving
    # the aspect ratio.
    height = 50
    def read_img(f):
        im = cv.imread(str(images_dir / f), cv.IMREAD_GRAYSCALE)
        if im is not None:
            im = im/255
            im = im[:,:,np.newaxis]
            if resize:
                im = tf.image.resize(im, [height, 10000],
                                     preserve_aspect_ratio=True)
        return im

    new_df['image'] = new_df['filename'].apply(read_img)

    # Remove images that couldn't be read
    new_df = new_df[new_df['image'].apply(lambda x: x is not None)]

    return new_df
