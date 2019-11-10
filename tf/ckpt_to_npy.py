"""Clean a ckpt to remove optimizer states."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import six
from six.moves import zip

import tensorflow as tf
import numpy as np

flags.DEFINE_string("input_ckpt", "",
                    help="input ckpt for cleaning")
flags.DEFINE_string("output_model_dir", "./",
                    help="output dir for cleaned ckpt")

FLAGS = flags.FLAGS


def clean_ckpt(_):
  """Core function."""
  input_ckpt = FLAGS.input_ckpt
  output_model_dir = FLAGS.output_model_dir
  output_file = os.path.join(output_model_dir, 'model.npy')
  model = {}
  tf.reset_default_graph()

  tf.logging.info("Loading from %s", input_ckpt)
  var_list = tf.contrib.framework.list_variables(input_ckpt)
  reader = tf.contrib.framework.load_checkpoint(input_ckpt)
  var_values, var_dtypes = {}, {}

  for (name, _) in var_list:
    if name.startswith("global_step") or "adam" in name.lower():
      continue
    tensor = reader.get_tensor(name)

    var_dtypes[name] = tensor.dtype
    var_values[name] = tensor

    model[name] = {'dtype': tensor.dtype, 'weight': tensor}
  np.save(output_file, model)

if __name__ == "__main__":
  tf.app.run(clean_ckpt)
