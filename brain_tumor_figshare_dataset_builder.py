"""brain_tumor_figshare dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for brain_tumor_figshare dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(brain_tumor_figshare): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(512, 512, 3)),
            'label': tfds.features.ClassLabel(names=['meningioma', 'glioma', 'pituitary tumor']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://doi.org/10.6084/m9.figshare.1512427.v5',
    )

  # https://www.tensorflow.org/datasets/add_dataset#specifying_dataset_splits
  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('http://127.0.0.1:8000/figshare.zip')
    # path = dl_manager.download_and_extract('https://assets.birkhoff.me/research/datasets/figshare.zip')

    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # 1/*.png is meningioma, 2/*.png is glioma, 3/*.png is pituitary tumor
    for f in path.glob('1/*.png'):
        yield str(f).split('/')[-1], {
            'image': f,
            'label': 'meningioma',
        }

    for f in path.glob('2/*.png'):
        yield str(f).split('/')[-1], {
            'image': f,
            'label': 'glioma',
        }

    for f in path.glob('3/*.png'):
        yield str(f).split('/')[-1], {
            'image': f,
            'label': 'pituitary tumor',
        }
