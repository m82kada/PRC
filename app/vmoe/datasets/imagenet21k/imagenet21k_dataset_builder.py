"""imagenet21k dataset."""

import tensorflow_datasets as tfds
import io
from pathlib import Path

CMYK_IMAGES = [
    'n01739381_1309.JPEG',
    'n02077923_14822.JPEG',
    'n02447366_23489.JPEG',
    'n02492035_15739.JPEG',
    'n02747177_10752.JPEG',
    'n03018349_4028.JPEG',
    'n03062245_4620.JPEG',
    'n03347037_9675.JPEG',
    'n03467068_12171.JPEG',
    'n03529860_11437.JPEG',
    'n03544143_17228.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n03961711_5286.JPEG',
    'n04033995_2932.JPEG',
    'n04258138_17003.JPEG',
    'n04264628_27969.JPEG',
    'n04336792_7448.JPEG',
    'n04371774_5854.JPEG',
    'n04596742_4225.JPEG',
    'n07583066_647.JPEG',
    'n13037406_4650.JPEG',
]

PNG_IMAGES = ['n02105855_2933.JPEG']

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for imagenet21k dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  """
  def _fix_image(self, image_fname, image):
    """Fix image color system and format starting from v 3.0.0."""
    if self.version < '3.0.0':
      return image
    if image_fname in CMYK_IMAGES:
      image = io.BytesIO(tfds.core.utils.jpeg_cmyk_to_rgb(image.read()))
    elif image_fname in PNG_IMAGES:
      image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()))
    return image

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(imagenet21k): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...

            'image': tfds.features.Image(encoding_format='jpeg'),
            #'label': tfds.features.ClassLabel(names_file=21843),
            'label': tfds.features.ClassLabel(num_classes=21843),
            'file_name': tfds.features.Text(),
            #'image': tfds.features.Image(shape=(None, None, 3)),
            #'label': tfds.features.ClassLabel(num_classes=21843),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(imagenet21k): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(imagenet21k): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        #'train': self._generate_examples(dl_manager.extract('/home/user/tensorflow_datasets/downloads/manual/winter21_whole.tar.gz')),
        'train': self._generate_examples(dl_manager.iter_archive('/home/user/tensorflow_datasets/downloads/manual/winter21_whole.tar.gz')),
    }

  def _generate_examples(self, archive):
    """Yields examples."""
    label_to_id = {}
    with open("/home/user/tensorflow_datasets/downloads/manual/imagenet21k_wordnet_ids.txt") as f:
      l = f.readlines()
      for i in l:
        label = i.replace("\r\n", "\n").split("\n")[0]
        if label == "":
          continue
        label_to_id[label] = len(label_to_id)

    for fname, fobj in archive:
      label = label_to_id[fname[15:-4]]
      fobj_mem = io.BytesIO(fobj.read())
      for image_fname, image in tfds.download.iter_archive(fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
        image = self._fix_image(image_fname, image)
        yield image_fname, {
            'image': image,
            'label': label,
            'file_name': image_fname,
        }
