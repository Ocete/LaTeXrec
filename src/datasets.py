from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf
import linecache
from ambiguities import remove_ambiguities

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
                 remove_amb,
                 shuffle_seed=123456,
                 test_fraction=0.1,
                 force_reloading=False, 
                 im2latex_configuration=False):
    
    """
    Load a dataset.

    Params:
    - base_path: directory in which it is contained.
    - images_dir: directory containing the images.
    - filename2line: path to file containing the filename -> line in formulas
      file mapping.
    - formulas_file: path to file containing the list of formulas.
    - remove_amb: Whether to remove ambiguities from the dataset.
    - shuffle_seed: seed used to shuffle the dataframe after it has been read.
    - test_fraction: fraction of data used for test vs (train + validation)
    - force_reloading: If true, force to recompute the whole dataset instead
      of loading a cached version.
    - im2latex_configuration: whether to use the configuration for the im2latex
      datasets.
    """
    
    # If the dataset was previously computed, load it. Else, generate it.
    if not force_reloading and Path(base_path / 'filename_formulas_df.json').exists():
        dataset_df = pd.read_json(base_path / 'filename_formulas_df.json')
        
    else:
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
        fn2l_df = fn2l_df.drop(['line'], axis=1)

        if remove_amb:
            fn2l_df['formula'] = fn2l_df['formula'].apply(remove_ambiguities)

        # Shuffle the dataset using the given seed
        dataset_df = fn2l_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    
        # Write the data in memory so we don't have to precompute it every time
        dataset_df.to_json(base_path / 'filename_formulas_df.json', default_handler=str)
    
    # Divide the data in (training + validation) and test
    j = int( (1.0 - test_fraction) * dataset_df.shape[0] )
    train_df, test_df = dataset_df[:j], dataset_df[j:] 
    
    return train_df, test_df.reset_index(drop=True)


class LaTeXrecDataset(tf.data.Dataset):
    def read_img(images_dir, file_name):
        height = 50
        im = cv.imread(str(images_dir / file_name), cv.IMREAD_GRAYSCALE)
        if im is not None:
            im = im/255
            im = im[:,:,np.newaxis]
            im = tf.image.resize(im, [height, 10000], preserve_aspect_ratio=True)            
        return im
    
    def _generator(cls, df, images_dir, tokenized_formulas):
        for index, row in df.iterrows():
            img = cls.read_img(images_dir, row['filename'])
            token_seq = tokenized_formulas[index]
            
            imgs_generator = lambda: iter([img].apply(lambda x: tf.cast(x, dtype=tf.float16)))
            dataset_img = tf.data.Dataset.from_generator(
                imgs_generator, output_types=tf.float16).map(lambda x: x) # Convert to Tensor
            
            dataset_seq = tf.data.Dataset.from_tensor_slices([token_seq]).map(lambda x: x) # Convert to Tensor

            yield (img, token_seq)

    def __new__(cls, df, images_dir):
        # Train the tokenizer and precompute all the tokenized formulas
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&:;?@`~ ')
        tokenizer.fit_on_texts(df['formula'])
        tokenized_formulas = tokenizer.texts_to_sequences(df['formula'])
        tokenized_formulas = list(map(lambda x: [len(tokenizer.word_index)] + x +
                              [len(tokenizer.word_index)+1], tokenized_formulas))
        tokenized_formulas = tf.ragged.constant(tokenized_formulas)
        
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(cls, df, images_dir, tokenized_formulas),
            output_types=(tf.Tensor, tf.Tensor),
            # output_shapes=(1,)
            # args=(df, images_dir, tokenized_formulas)
        )


# How to use the LaTeXrecDataset class
image_dir = get_paths(1)[1]
ds = LaTeXrecDataset(train_df[0:10], image_dir)
for (img, form) in ds:
    print(img.shape, ds)