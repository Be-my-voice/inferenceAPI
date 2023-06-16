import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np

import tensorflow as tf
import keras
from keras import layers

import zipfile
import os
import shutil

import mediapipe as mp
import json

print(tf.__version__)

def list_files_from_zip_path(zip_file_path):
    """ 
    List the files in each class of the dataset given a local zip file path.

    Args:
      zip_file_path: Local file path of the zip file.

    Returns:
      List of files in each of the classes.
    """
    files = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def get_class(fname):
  """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Return:
      Class that the file belongs to.
  """
  return fname.split('_')[0]

def get_files_per_class(files):
  """
    Retrieve the files that belong to each class. 

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class

def download_from_zip(zip_file_path, to_dir, file_names):
    """ 
    Download the contents of the zip file from the local zip file path.

    Args:
      zip_file_path: Local file path of the zip file.
      to_dir: A directory to download data to.
      file_names: Names of files to download.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip:
        for fn in tqdm.tqdm(file_names):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn

            fn = pathlib.Path(fn).parts[-1]
            output_file = to_dir / class_name / fn
            unzipped_file.rename(output_file)

def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.
    
    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def download_ucf_101_subset(zip_file_path, num_classes, splits, download_dir):
    """ 
    Download a subset of the UCF101 dataset and split them into various parts,
    such as training, validation, and test.

    Args:
      zip_file_path: Local file path of the ZIP file with the data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
              (value is the number of files per split).
      download_dir: Directory to download data to.

    Returns:
      Mapping of the directories containing the subsections of data.
    """
    files = list_files_from_zip_path(zip_file_path)
    
    files = [f for f in files if os.path.basename(f)]  # Remove items without filenames
    
    files_for_class = get_files_per_class(files)

    classes = list(files_for_class.keys())[:num_classes]
    for cls in classes:
        random.shuffle(files_for_class[cls])

    # Only use the number of classes you want in the dictionary
    files_for_class = {x: files_for_class[x] for x in classes}

    dirs = {}
    for split_name, split_count in splits.items():
        print(split_name, ":")
        split_dir = pathlib.Path(download_dir) / split_name
        split_files, files_for_class = split_class_lists(files_for_class, split_count)
        download_from_zip(zip_file_path, split_dir, split_files)
        dirs[split_name] = split_dir

    return dirs

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.
    
    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 4):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  
  print(video_path)

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))
    

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
    
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])
  
  def call(self, x):
    return self.seq(x)

class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])
    
  def call(self, x):
    return self.seq(x)

class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)
  
  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  
      
      Args:
        video: Tensor representation of the video, in the form of a set of frames.
      
      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos
  
class VideoFrameGenerator:
  def __init__(self, path, n_frames):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames

  def __call__(self):
    video_frames = frames_from_video_file(self.path, self.n_frames)
    yield video_frames
    

FILEPATH = 'dataset.zip'
# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224

class InferModel:
    def __init__(self):
        shutil.rmtree('./dataset/')
        download_dir = pathlib.Path('./dataset/')
        subset_paths = download_ucf_101_subset(FILEPATH, 
                                num_classes = 4, 
                                splits = {"train": 2, "val": 1, "test": 0},
                                download_dir = download_dir)



        self.n_frames = 16
        self.batch_size = 8

        output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                            tf.TensorSpec(shape = (), dtype = tf.int16))

        val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], self.n_frames),
                                                output_signature = output_signature)

        fg = FrameGenerator(subset_paths['train'], self.n_frames, training=True)
        self.labels = list(fg.class_ids_for_name.keys())
        print(self.labels)

        input_shape = (None, 16, HEIGHT, WIDTH, 3)
        input_ = layers.Input(shape=(input_shape[1:]))
        x = input_

        x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

        # Block 1
        x = add_residual_block(x, 16, (3, 3, 3))
        x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

        # Block 2
        x = add_residual_block(x, 32, (3, 3, 3))
        x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

        # Block 3
        x = add_residual_block(x, 64, (3, 3, 3))
        x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

        # Block 4
        x = add_residual_block(x, 128, (3, 3, 3))

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10)(x)

        self.model = keras.Model(input_, x)

        print('compiling model...')
        self.model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                    optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
                    metrics = ['accuracy'])

        print('loading model...')
        self.model.load_weights('./weights/ssl')

        self.output_signature_inference = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32))

    def predict(self, filename):
        try:
            inference_video = tf.data.Dataset.from_generator(VideoFrameGenerator(filename, self.n_frames),
                                                        output_signature = self.output_signature_inference)
            inference_video = inference_video.batch(self.batch_size)

            frames = next(iter(inference_video))
            predicted = self.model.predict(inference_video, verbose=1)
            predicted = tf.concat(predicted, axis=0)
            # print(predicted)
                # tf.sort(predicted, axis=0, direction='DESCENDING', name=None)
            print(tf.sort(tf.reshape(predicted, [10]), axis=0, direction='DESCENDING', name=None))

            predicted = tf.argmax(predicted, axis=1)
            return self.labels[predicted[0]]
        except:
           print("An error occured")
           return "Error"


# inf = InferModel()
# inf.predict('break_101')    

       
    