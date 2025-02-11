# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of data processing ops.

All ops should return data processing functors. Data examples are represented
as a dictionary of tensors.

Most of these were originally implemented by: Lucas Beyer, Alex Kolesnikov,
Xiaohua Zhai and other collaborators from Google Brain Zurich.
"""
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import random

#try:
#  from cloud_tpu.models.efficientnet import autoaugment  # pylint: disable=g-import-not-at-top
#except ImportError:
#  autoaugment = None

from vmoe.data import autoaugment
from vmoe.data import autoaugment_R


VALID_KEY = '__valid__'


class InKeyOutKey(object):
  """Decorator for preprocessing ops, which adds `inkey` and `outkey` arguments.

  Note: Only supports single-input single-output ops.
  """

  def __init__(self, indefault='image', outdefault='image'):
    self.indefault = indefault
    self.outdefault = outdefault

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args, key=None,
                       inkey=self.indefault, outkey=self.outdefault, **kw):
      orig_pp_fn = orig_get_pp_fn(*args, **kw)
      def _ikok_pp_fn(data):
        data[key or outkey] = orig_pp_fn(data[key or inkey])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn


@InKeyOutKey()
def central_crop(crop_size):
  """Makes central crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively.

  Returns:
    A function, that applies central crop.
  """
  if isinstance(crop_size, int):
    crop_size = (crop_size, crop_size)
  crop_size = tuple(crop_size)

  def _crop(image):
    h, w = crop_size[0], crop_size[1]
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

  return _crop


def copy(inkey, outkey):
  """Copies value of `inkey` into `outkey`."""

  def _copy(data):
    data[outkey] = data[inkey]
    return data

  return _copy


@InKeyOutKey()
def decode(channels=3):
  """Decodes an encoded image string, see tf.io.decode_image."""

  def _decode(image):
    return tf.io.decode_image(image, channels=channels, expand_animations=False)

  return _decode


@InKeyOutKey()
def decode_jpeg_and_inception_crop(resize_size=None, area_min=5, area_max=100):
  """Decodes jpeg string and makes inception-style image crop.

  See `inception_crop` for details.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)

    if resize_size:
      image = resize(resize_size)({'image': image})['image']

    return image

  return _inception_crop


@InKeyOutKey()
def flip_lr():
  """Flips an image horizontally with probability 50%."""

  def _random_flip_lr_pp(image):
    return tf.image.random_flip_left_right(image)

  return _random_flip_lr_pp


@InKeyOutKey()
def gaussian_blur_image(size):
  def _gaussian(image):
    image = tf2.image.convert_image_dtype(image, dtype=tf2.float32)
    sigma = tf2.random.uniform([], 0.1, 2.0, dtype=tf2.float32)
    image = gaussian_blur(
        image, kernel_size=size//10, sigma=sigma, padding='SAME')
    return tf2.image.convert_image_dtype(image, dtype=tf2.uint8) # image
  return _gaussian

def ignore_no_labels(labels_key='labels', valid_key=VALID_KEY):

  def _transform(data):
    valid = tf.math.logical_and(tf.cast(data.get(valid_key, True), tf.bool),
                                tf.size(data[labels_key]) > 0)
    return data | {valid_key: valid}

  return _transform


@InKeyOutKey()
def inception_crop(resize_size=None, area_min=5, area_max=100,
                   resize_method='bilinear'):
  """Makes inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    resize_method: rezied method, see tf.image.resize docs for options.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image):  # pylint: disable=missing-docstring
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    if resize_size:
      crop = resize(resize_size, resize_method)({'image': crop})['image']
    return crop

  return _inception_crop


def keep(*keys):
  """Keeps only the given keys (in addition to metadata keys)."""
  keys = tuple(keys) + (VALID_KEY,)

  def _keep(data):
    return {k: v for k, v in data.items() if k in keys}

  return _keep


@InKeyOutKey(indefault='labels', outdefault='labels')
def onehot(depth, multi=True, on=1.0, off=0.0):
  """One-hot encodes the input.

  Args:
    depth: Length of the one-hot vector (how many classes).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).

  Returns:
    Data dictionary.
  """

  def _onehot(label):
    # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    if label.shape.rank > 0 and multi:
      x = tf.scatter_nd(label[:, None],
                        tf.ones(tf.shape(label)[0]), (depth,))
      x = tf.clip_by_value(x, 0, 1) * (on - off) + off
    else:
      x = tf.one_hot(label, depth, on_value=on, off_value=off)
    return x

  return _onehot


@InKeyOutKey()
def randaug(num_layers: int = 2, magnitude: int = 10):
  """Creates a function that applies RandAugment.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].

  Returns:
    a function that applies RandAugment.
  """
  #if autoaugment is None:
  #  raise ValueError(
  #      "In order to use RandAugment you need to install the 'cloud_tpu' "
  #      "package. Clone the https://github.com/tensorflow/tpu repository, "
  #      "name it 'cloud_tpu', and add the corresponding directory to your "
  #      "PYTHONPATH.")

  ra = autoaugment.RandAugment(num_layers, magnitude)
  def _randaug(image):
    return ra.distort(image)
    #return autoaugment.distort_image_with_randaugment(
    #    image, num_layers, magnitude)

  return _randaug


@InKeyOutKey()
def resize(resize_size, resize_method='bilinear'):
  """Resizes image to a given size.

  Args:
    resize_size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.
    resize_method: rezied method, see tf.image.resize docs for options.

  Returns:
    A function for resizing an image.

  """
  if isinstance(resize_size, int):
    resize_size = (resize_size, resize_size)
  resize_size = tuple(resize_size)

  def _resize(image):
    # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
    # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
    # In particular it was not equivariant with rotation and lead to the network
    # to learn a shortcut in self-supervised rotation task, if rotation was
    # applied after resize.
    dtype = image.dtype
    image = tf2.image.resize(image, resize_size, resize_method)
    return tf.cast(image, dtype)

  return _resize


@InKeyOutKey()
def resize_small(smaller_size):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """

  def _resize_small(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # Figure out the necessary h/w.
    ratio = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

    return tf.image.resize_area(image[None], [h, w])[0]

  return _resize_small


@InKeyOutKey()
def reshape(new_shape):
  """Reshapes image to a given new shape.

  Args:
    new_shape: new shape size (h, w, c).

  Returns:
    A function for reshaping an image.

  """

  def _reshape(image):
    """Reshapes image to a given size."""
    dtype = image.dtype
    image = tf.reshape(image, new_shape)
    return tf.cast(image, dtype)

  return _reshape


@InKeyOutKey()
def value_range(vmin, vmax, in_min=0, in_max=255.0, clip_values=False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.

  Returns:
    A function to rescale the values.
  """

  def _value_range(image):
    """Scales values in given range."""
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if clip_values:
      image = tf.clip_by_value(image, vmin, vmax)
    return image

  return _value_range

#def get_min_max(y, x, window, patch_len):
#  # 引数はpatch座標系のy, x
#  # 返り値は元の画像座標系の 右下のy, x, 左上のy, x
#  def get_original_coord(num, is_x=0): # is_x: True -> 1
#    return (window[2 + is_x] - window[is_x]) * tf2.cast(num, tf2.float32) / tf2.cast(tf2.constant(patch_len), tf2.float32) + window[is_x]
#  tmp1 = tf2.stack([get_original_coord(y), get_original_coord(y+1)])
#  tmp2 = tf2.stack([get_original_coord(x, 1), get_original_coord(x+1, 1)])
#  return tf2.reduce_max(tmp1), tf2.reduce_max(tmp2), tf2.reduce_min(tmp1), tf2.reduce_min(tmp2)

def decode_and_ssl_augment(resize_size=None, original_image_size=384, patch_size=32, area_min=5, area_max=70, area2_min=60, area2_max=100, use_cls=1, channels=3):
  """Decodes jpeg string and makes inception-style image crop.

  See `inception_crop` for details.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.

  Returns:
    A function, that applies inception crop.
  """
  image_size = int(resize_size if resize_size else original_image_size)
  patch_len = int(image_size // patch_size)
  def _dataaug(data):  # pylint: disable=missing-docstring
    _image = tf.io.decode_image(data['image'], channels=channels, expand_animations=False)
    tmp = {}
    for i in range(2):
      begin, size, _ = tf2.image.sample_distorted_bounding_box(
          tf.shape(_image),
          tf2.zeros([0, 0, 4], tf2.float32),
          #area_range=(area_min / 100, area_max / 100),
          area_range=(area_min / 100, area_max / 100) if i == 0 else (area2_min / 100, area2_max / 100),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)

      image = tf.slice(_image, begin, size)
      image.set_shape([None, None, _image.shape[-1]])
      if resize_size:
        image = resize(resize_size)({'image': image})['image']

      offset_y, offset_x, _ = tf2.unstack(begin)
      target_height, target_width, _ = tf2.unstack(size)

      # flip
      #if tf2.random.uniform([]) < 0.5:
      p = tf2.random.uniform([])
      image = tf2.cond(p < 0.5, lambda: image, lambda: tf2.image.flip_left_right(image))
      #tmp[f'window_{i+1}'] = tf2.cond(p < 0.5,
      #                                lambda: tf2.cast(tf2.stack([offset_y, offset_x, offset_y + target_height, offset_x + target_width]), dtype=tf2.float32),
      #                                lambda: tf2.cast(tf2.stack([offset_y, offset_x + target_width, offset_y + target_height, offset_x]), dtype=tf2.float32))

      # crop 座標系 -> original 座標系
      offset_x = tf2.cast(offset_x, dtype=tf2.float32)
      offset_y = tf2.cast(offset_y, dtype=tf2.float32)
      target_width = tf2.cast(target_width, dtype=tf2.float32)
      target_height = tf2.cast(target_height, dtype=tf2.float32)
      tmp[f'R_{i+1}'] = tf2.cond(p < 0.5,
                                 lambda: tf2.stack([
                                   [target_height / resize_size, 0., offset_y],
                                   [0., target_width / resize_size, offset_x],
                                   [0., 0., 1.],
                                 ]),
                                 lambda: tf2.stack([
                                   [target_height / resize_size, 0., offset_y],
                                   [0., -target_width / resize_size, offset_x + target_width],
                                   [0., 0., 1.],
                                 ]))

      tmp[f'R2_{i+1}'] = tf2.cond(p < 0.5,
                                 lambda: tf2.stack([
                                   [resize_size / target_height, 0., -resize_size * offset_y / target_height],
                                   [0., resize_size / target_width, -resize_size * offset_x / target_width],
                                   [0., 0., 1.],
                                 ]),
                                 lambda: tf2.stack([
                                   [resize_size / target_height, 0., -resize_size * offset_y / target_height],
                                   [0., -resize_size / target_width, resize_size * (target_width + offset_x) / target_width],
                                   [0., 0., 1.],
                                 ]))

      tmp[f'S_{i+1}'] = target_height * target_width
      # gaussian blur
      image = tf2.image.convert_image_dtype(image, dtype=tf2.float32)
      #with tf2.device('/cpu:0'):
      #  # tf2.image.random_hue内でnanが発生してエラーが出る．下記のurlの方法に従い対処
      #  # https://stackoverflow.com/questions/40790009/tensorflow-generate-random-values-unexpected-behaviour
      #  image = random_blur(image, resize_size if resize_size else target_height)
      #  image = color_jitter(image, 0.3)
      tmp[f'image_{i+1}'] = tf2.image.convert_image_dtype(image, dtype=tf2.uint8) # image
    # image1のpatch size < image2のpatch sizeとする
    #data['image'] = tf2.stack([tmp['image_1'], tmp['image_2']])
    #data['windows'] = tf2.stack([tmp['window_1'], tmp['window_2']])
    rev_flag = tmp['S_1'] < tmp['S_2']
    data['image'] = tf2.cond(rev_flag, lambda: tf2.stack([tmp['image_1'], tmp['image_2']]), lambda: tf2.stack([tmp['image_2'], tmp['image_1']]))
    data['R'] = tf2.cond(rev_flag,
                         lambda: tf2.matmul(tmp['R2_2'], tmp['R_1']),
                         lambda: tf2.matmul(tmp['R2_1'], tmp['R_2'])
                )
    return data

  return _dataaug

def decode_jpeg_and_ssl_augment(resize_size=None, original_image_size=384, patch_size=32, area_min=5, area_max=70, area2_min=60, area2_max=100, use_cls=1):
#def decode_jpeg_and_ssl_augment(resize_size=None, original_image_size=384, patch_size=32, area_min=5, area_max=100, area2_min=60, area2_max=100, use_cls=1):
  """Decodes jpeg string and makes inception-style image crop.

  See `inception_crop` for details.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.

  Returns:
    A function, that applies inception crop.
  """
  image_size = int(resize_size if resize_size else original_image_size)
  patch_len = int(image_size // patch_size)
  def _dataaug(data):  # pylint: disable=missing-docstring
    shape = tf2.image.extract_jpeg_shape(data['image'])
    tmp = {}
    #_offset_y, _offset_x, _ = tf2.unstack(begin)
    #_target_height, _target_width, _ = tf2.unstack(size)
    #_offset_y = tf2.cast(_offset_y, tf2.float32)
    #_offset_x = tf2.cast(_offset_x, tf2.float32)
    #_target_height = tf2.cast(_target_height, tf2.float32)
    #_target_width = tf2.cast(_target_width, tf2.float32)
    #for i in range(2):
    #  target_height = tf2.cast((area2_min + (100 - area2_min) * tf2.random.uniform([])) / 100 * _target_height, tf2.int32)
    #  target_width = tf2.cast((area2_min + (100 - area2_min) * tf2.random.uniform([])) / 100 * _target_width, tf2.int32)
    #  offset_y = tf2.cast(_offset_y + (_target_height - tf2.cast(target_height, tf2.float32)) * tf2.random.uniform([]), tf2.int32)
    #  offset_x = tf2.cast(_offset_x + (_target_width - tf2.cast(target_width, tf2.float32)) * tf2.random.uniform([]), tf2.int32)
    #  crop_window = tf2.stack([offset_y, offset_x, target_height, target_width])
    for i in range(2):
      begin, size, _ = tf2.image.sample_distorted_bounding_box(
          shape,
          tf2.zeros([0, 0, 4], tf2.float32),
          area_range=(area_min / 100, area_max / 100) if i == 0 else (area2_min / 100, area2_max / 100),
          #area_range=(area_min / 100, area_max / 100),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)

      offset_y, offset_x, _ = tf2.unstack(begin)
      target_height, target_width, _ = tf2.unstack(size)
      crop_window = tf2.stack([offset_y, offset_x, target_height, target_width])

      image = tf2.image.decode_and_crop_jpeg(data['image'], crop_window, channels=3)

      if resize_size:
        image = resize(resize_size)({'image': image})['image']

      # flip
      #if tf2.random.uniform([]) < 0.5:
      p = tf2.random.uniform([])
      image = tf2.cond(p < 0.5, lambda: image, lambda: tf2.image.flip_left_right(image))
      #tmp[f'window_{i+1}'] = tf2.cond(p < 0.5,
      #                                lambda: tf2.cast(tf2.stack([offset_y, offset_x, offset_y + target_height, offset_x + target_width]), dtype=tf2.float32),
      #                                lambda: tf2.cast(tf2.stack([offset_y, offset_x + target_width, offset_y + target_height, offset_x]), dtype=tf2.float32))

      # crop 座標系 -> original 座標系
      offset_x = tf2.cast(offset_x, dtype=tf2.float32)
      offset_y = tf2.cast(offset_y, dtype=tf2.float32)
      target_width = tf2.cast(target_width, dtype=tf2.float32)
      target_height = tf2.cast(target_height, dtype=tf2.float32)
      tmp[f'R_{i+1}'] = tf2.cond(p < 0.5,
                                 lambda: tf2.stack([
                                   [target_height / resize_size, 0., offset_y],
                                   [0., target_width / resize_size, offset_x],
                                   [0., 0., 1.],
                                 ]),
                                 lambda: tf2.stack([
                                   [target_height / resize_size, 0., offset_y],
                                   [0., -target_width / resize_size, offset_x + target_width],
                                   [0., 0., 1.],
                                 ]))

      tmp[f'R2_{i+1}'] = tf2.cond(p < 0.5,
                                 lambda: tf2.stack([
                                   [resize_size / target_height, 0., -resize_size * offset_y / target_height],
                                   [0., resize_size / target_width, -resize_size * offset_x / target_width],
                                   [0., 0., 1.],
                                 ]),
                                 lambda: tf2.stack([
                                   [resize_size / target_height, 0., -resize_size * offset_y / target_height],
                                   [0., -resize_size / target_width, resize_size * (target_width + offset_x) / target_width],
                                   [0., 0., 1.],
                                 ]))

      tmp[f'S_{i+1}'] = target_height * target_width
      data[f'S_{i+1}'] = target_height * target_width
      # gaussian blur
      image = tf2.image.convert_image_dtype(image, dtype=tf2.float32)
      #with tf2.device('/cpu:0'):
      #  # tf2.image.random_hue内でnanが発生してエラーが出る．下記のurlの方法に従い対処
      #  # https://stackoverflow.com/questions/40790009/tensorflow-generate-random-values-unexpected-behaviour
      #  image = random_blur(image, resize_size if resize_size else target_height)
      #  image = color_jitter(image, 0.3)
      tmp[f'image_{i+1}'] = tf2.image.convert_image_dtype(image, dtype=tf2.uint8) # image
    # image1のpatch size < image2のpatch sizeとする
    #data['image'] = tf2.stack([tmp['image_1'], tmp['image_2']])
    #data['windows'] = tf2.stack([tmp['window_1'], tmp['window_2']])
    rev_flag = tmp['S_1'] < tmp['S_2']
    data['image'] = tf2.cond(rev_flag, lambda: tf2.stack([tmp['image_1'], tmp['image_2']]), lambda: tf2.stack([tmp['image_2'], tmp['image_1']]))
    data['R'] = tf2.cond(rev_flag,
                         lambda: tf2.matmul(tmp['R2_2'], tmp['R_1']),
                         lambda: tf2.matmul(tmp['R2_1'], tmp['R_2'])
                )
    return data

  return _dataaug

def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf2.cond(
      tf2.less(
          tf2.random.uniform([], minval=0, maxval=1, dtype=tf2.float32),
          tf2.cast(p, tf2.float32)), lambda: func(x), lambda: x)


def random_brightness(image, max_delta, impl='simclrv2'):
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf2.random.uniform([], tf2.maximum(1.0 - max_delta, 0),
                               1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = tf2.image.random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError('Unknown impl {} for random brightness.'.format(impl))
  return image


def to_grayscale(image, keep_channels=True):
  image = tf2.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf2.tile(image, [1, 1, 3])
  return image


def color_jitter(image, strength, random_order=True, impl='simclrv2'):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(
        image, brightness, contrast, saturation, hue, impl=impl)
  else:
    return color_jitter_nonrand(
        image, brightness, contrast, saturation, hue, impl=impl)


def color_jitter_nonrand(image,
                         brightness=0,
                         contrast=0,
                         saturation=0,
                         hue=0,
                         impl='simclrv2'):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  with tf2.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness, impl=impl)
      elif contrast != 0 and i == 1:
        x = tf2.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf2.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf2.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf2.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0,
                      impl='simclrv2'):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  with tf2.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness, impl=impl)

      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf2.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf2.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf2.image.random_hue(x, max_delta=hue)
      x = tf2.cond(tf2.less(i, 2),
                  lambda: tf2.cond(tf2.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf2.cond(tf2.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf2.random.shuffle(tf2.range(4))
    for i in range(4):
      image_tmp = apply_transform(perm[i], image)
      image = tf2.clip_by_value(image, 0., 1.)
    return image

def random_blur(image, height, p=1.0):
  """Randomly blur an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  def _transform(image):
    sigma = tf2.random.uniform([], 0.1, 2.0, dtype=tf2.float32)
    return gaussian_blur(
        image, kernel_size=height//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.


  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf2.cast(kernel_size / 2, dtype=tf2.int32)
  kernel_size = radius * 2 + 1
  x = tf2.cast(tf2.range(-radius, radius + 1), dtype=tf2.float32)
  blur_filter = tf2.exp(-tf2.pow(x, 2.0) /
                       (2.0 * tf2.pow(tf2.cast(sigma, dtype=tf2.float32), 2.0)))
  blur_filter /= tf2.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf2.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf2.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf2.shape(image)[-1]
  blur_h = tf2.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf2.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf2.expand_dims(image, axis=0)
  blurred = tf2.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf2.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf2.squeeze(blurred, axis=0)
  return blurred

#def decode_jpeg_and_randaug_ssl(resize_size=None, num_layers=2, magnitude=10, original_image_size=384, patch_size=32, area_min=5, area_max=100, use_cls=1):
def decode_jpeg_and_randaug_ssl(resize_size=None, num_layers=2, magnitude=10, original_image_size=384, patch_size=32, area_min=5, area_max=70, area2_min=60, area2_max=100, use_cls=1):
  """Decodes jpeg string and makes inception-style image crop.

  See `inception_crop` for details.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.

  Returns:
    A function, that applies inception crop.
  """
  ra = autoaugment_R.RandAugment(num_layers, magnitude)
  image_size = int(resize_size if resize_size else original_image_size)
  patch_len = int(image_size // patch_size)
  def _dataaug(data):  # pylint: disable=missing-docstring
    shape = tf2.image.extract_jpeg_shape(data['image'])
    tmp = {}
    for i in range(2):
      begin, size, _ = tf2.image.sample_distorted_bounding_box(
          shape,
          tf2.zeros([0, 0, 4], tf2.float32),
          #area_range=(area_min / 100, area_max / 100),
          area_range=(area_min / 100, area_max / 100) if i == 0 else (area2_min / 100, area2_max / 100),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)

      offset_y, offset_x, _ = tf2.unstack(begin)
      target_height, target_width, _ = tf2.unstack(size)
      crop_window = tf2.stack([offset_y, offset_x, target_height, target_width])

      image = tf2.image.decode_and_crop_jpeg(data['image'], crop_window, channels=3)

      if resize_size:
        image = resize(resize_size)({'image': image})['image']

      # flip
      p = tf2.random.uniform([])
      image = tf2.cond(p < 0.5, lambda: image, lambda: tf2.image.flip_left_right(image))

      # crop 座標系 -> original 座標系
      offset_x = tf2.cast(offset_x, dtype=tf2.float32)
      offset_y = tf2.cast(offset_y, dtype=tf2.float32)
      target_width = tf2.cast(target_width, dtype=tf2.float32)
      target_height = tf2.cast(target_height, dtype=tf2.float32)
      R = tf2.cond(p < 0.5,
                   lambda: tf2.stack([
                     [resize_size / target_height, 0., -resize_size * offset_y / target_height],
                     [0., resize_size / target_width, -resize_size * offset_x / target_width],
                     [0., 0., 1.],
                   ]),
                   lambda: tf2.stack([
                     [resize_size / target_height, 0., -resize_size * offset_y / target_height],
                     [0., -resize_size / target_width, resize_size * (target_width + offset_x) / target_width],
                     [0., 0., 1.],
                   ]))

      tmp[f'S_{i+1}'] = target_height * target_width
      image, R2 = ra.distort(image)
      tmp[f'image_{i+1}'] = image
      tmp[f'R_{i+1}'] = tf2.matmul(R2, R)
      #image = tf2.image.convert_image_dtype(image, dtype=tf2.float32)
      #tmp[f'image_{i+1}'] = tf2.image.convert_image_dtype(image, dtype=tf2.uint8) # image
    rev_flag = tmp['S_1'] < tmp['S_2']
    data['image'] = tf2.cond(rev_flag, lambda: tf2.stack([tmp['image_1'], tmp['image_2']]), lambda: tf2.stack([tmp['image_2'], tmp['image_1']]))
    data['R'] = tf2.cond(rev_flag,
                         lambda: tf2.matmul(tmp['R_2'], inv_R3(tmp['R_1'])), # tf.linalg.inv
                         lambda: tf2.matmul(tmp['R_1'], inv_R3(tmp['R_2']))
                )
    return data

  return _dataaug

#def randaug_ssl_mixup(resize_size=None, num_layers=2, magnitude=10, original_image_size=384, patch_size=32, area_min=5, area_max=100, area2_min=60, area2_max=100, use_cls=1):
#  """Decodes jpeg string and makes inception-style image crop.
#
#  See `inception_crop` for details.
#
#  Args:
#    resize_size: Resize image to this size after crop.
#    area_min: minimal crop area.
#    area_max: maximal crop area.
#
#  Returns:
#    A function, that applies inception crop.
#  """
#  ra = autoaugment_R.RandAugment(num_layers, magnitude)
#  image_size = int(resize_size if resize_size else original_image_size)
#  patch_len = int(image_size // patch_size)
#  def _dataaug(data):  # pylint: disable=missing-docstring
#    shape = data['image'].shape
#    tmp = {}
#    for i in range(2):
#      begin, size, _ = tf2.image.sample_distorted_bounding_box(
#          shape,
#          tf2.zeros([0, 0, 4], tf2.float32),
#          #area_range=(area_min / 100, area_max / 100),
#          area_range=(area_min / 100, area_max / 100) if i == 0 else (area2_min / 100, area2_max / 100),
#          min_object_covered=0,  # Don't enforce a minimum area.
#          use_image_if_no_bounding_boxes=True)
#
#      offset_y, offset_x, _ = tf2.unstack(begin)
#      target_height, target_width, _ = tf2.unstack(size)
#      crop_window = tf2.stack([offset_y, offset_x, target_height, target_width])
#
#      image = tf2.image.crop_to_bounding_box(data['image'], offset_y, offset_x, target_height, target_width)
#      #image = tf2.image.decode_and_crop_jpeg(data['image'], crop_window, channels=3)
#
#      if resize_size:
#        image = resize(resize_size)({'image': image})['image']
#
#      # flip
#      p = tf2.random.uniform([])
#      image = tf2.cond(p < 0.5, lambda: image, lambda: tf2.image.flip_left_right(image))
#
#      # crop 座標系 -> original 座標系
#      offset_x = tf2.cast(offset_x, dtype=tf2.float32)
#      offset_y = tf2.cast(offset_y, dtype=tf2.float32)
#      target_width = tf2.cast(target_width, dtype=tf2.float32)
#      target_height = tf2.cast(target_height, dtype=tf2.float32)
#      R = tf2.cond(p < 0.5,
#                   lambda: tf2.stack([
#                     [resize_size / target_height, 0., -resize_size * offset_y / target_height],
#                     [0., resize_size / target_width, -resize_size * offset_x / target_width],
#                     [0., 0., 1.],
#                   ]),
#                   lambda: tf2.stack([
#                     [resize_size / target_height, 0., -resize_size * offset_y / target_height],
#                     [0., -resize_size / target_width, resize_size * (target_width + offset_x) / target_width],
#                     [0., 0., 1.],
#                   ]))
#
#      tmp[f'S_{i+1}'] = target_height * target_width
#      image, R2 = ra.distort(image)
#      tmp[f'image_{i+1}'] = image
#      tmp[f'R_{i+1}'] = tf2.matmul(R2, R)
#      #image = tf2.image.convert_image_dtype(image, dtype=tf2.float32)
#      #tmp[f'image_{i+1}'] = tf2.image.convert_image_dtype(image, dtype=tf2.uint8) # image
#    rev_flag = tmp['S_1'] < tmp['S_2']
#    data['image'] = tf2.cond(rev_flag, lambda: tf2.stack([tmp['image_1'], tmp['image_2']]), lambda: tf2.stack([tmp['image_2'], tmp['image_1']]))
#    data['R'] = tf2.cond(rev_flag,
#                         lambda: tf2.matmul(tmp['R_2'], inv_R3(tmp['R_1'])), # tf.linalg.inv
#                         lambda: tf2.matmul(tmp['R_1'], inv_R3(tmp['R_2']))
#                )
#    return data
#
#  return _dataaug

def inv_R3(R: tf.Tensor) -> tf.Tensor:
  k = (R[0,0] * R[1,1] * R[2,2] + R[0,1] * R[1,2] * R[2,0] + R[0,2] * R[1,0] * R[2,1]) - (R[0,2] * R[1,1] * R[2,0] + R[0,1] * R[1,0] * R[2,2] * R[0,0] * R[1,2] * R[2,1])
  return tf.reshape(tf.stack([
    (R[1,1] * R[2,2] - R[1,2] * R[2,1]), -(R[0,1] * R[2,2] - R[0,2] * R[2,1]), (R[0,1] * R[1,2] - R[0,2] * R[1,1]),
    -(R[1,0] * R[2,2] - R[1,2] * R[2,0]), (R[0,0] * R[2,2] - R[0,2] * R[2,0]), -(R[0,0] * R[1,2] - R[0,2] * R[1,0]),
    (R[1,0] * R[2,1] - R[1,1] * R[2,0]), -(R[0,0] * R[2,1] - R[0,1] * R[2,0]), (R[0,0] * R[1,1] - R[0,1] * R[1,0])
  ]), [3, 3]) / k
