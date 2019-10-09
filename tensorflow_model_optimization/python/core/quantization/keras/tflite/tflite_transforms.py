# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""TFLite transforms."""

from tensorflow.python import keras

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern
Transform = transforms.Transform

_ConvBatchNorm2D = conv_batchnorm._ConvBatchNorm2D  # pylint: disable=protected-access
_DepthwiseConvBatchNorm2D = conv_batchnorm._DepthwiseConvBatchNorm2D  # pylint: disable=protected-access


def _get_conv_bn_params(conv_layer, bn_layer, relu_layer=None):
  """Retrieve conv_bn params within wrapped layers."""
  unwrapped_conv = conv_layer['config']['layer']
  unwrapped_bn = bn_layer['config']['layer']

  if 'use_bias' in unwrapped_conv['config']:
    del unwrapped_conv['config']['use_bias']

  if 'name' in unwrapped_bn['config']:
    del unwrapped_bn['config']['name']

  # TODO(pulkitb): remove key conflicts
  conv_bn_params = dict(
      list(unwrapped_conv['config'].items()) +
      list(unwrapped_bn['config'].items()))

  if relu_layer is not None:
    unwrapped_relu_layer = relu_layer['config']['layer']
    conv_bn_params['post_activation'] = keras.layers.deserialize(
        unwrapped_relu_layer)

  return conv_bn_params


class Conv2DBatchNormFold(transforms.Transform):
  """Conv2DBatchNormFold."""

  def pattern(self):
    return LayerPattern('BatchNormalization', {},
                        [LayerPattern('Conv2D', {}, [])])

  def _get_fused_conv_bn_layer(self, match_layer, has_relu=False):
    if has_relu:
      relu_layer = match_layer.layer
      bn_layer_node = match_layer.input_layers[0]
      bn_layer = bn_layer_node.layer
      conv_layer = bn_layer_node.input_layers[0].layer
      return self._map_to_conv_bn_layer(conv_layer, bn_layer, relu_layer)

    bn_layer = match_layer.layer
    conv_layer = match_layer.input_layers[0].layer
    return self._map_to_conv_bn_layer(conv_layer, bn_layer)

  def _map_to_conv_bn_layer(self, conv_layer, bn_layer, relu_layer=None):
    conv_bn_params = _get_conv_bn_params(conv_layer, bn_layer, relu_layer)
    return _ConvBatchNorm2D(**conv_bn_params)

  def replacement(self, match_layer):
    conv_bn_layer = self._get_fused_conv_bn_layer(match_layer)

    layer_config = keras.layers.serialize(conv_bn_layer)
    layer_config['name'] = layer_config['config']['name']

    return LayerNode(layer_config, [])


class Conv2DBatchNormReLU6Fold(Conv2DBatchNormFold):
  """Conv2DBatchNormReLU6Fold."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6}, [
        LayerPattern('BatchNormalization', {}, [LayerPattern('Conv2D', {}, [])])
    ])

  def replacement(self, match_layer):
    conv_bn_layer = self._get_fused_conv_bn_layer(match_layer, has_relu=True)

    layer_config = keras.layers.serialize(conv_bn_layer)
    layer_config['name'] = layer_config['config']['name']

    return LayerNode(layer_config, [])

  def custom_objects(self):
    return {'_ConvBatchNorm2D': _ConvBatchNorm2D}


class DepthwiseConv2DBatchNormReLU6Fold(Conv2DBatchNormReLU6Fold):
  """DepthwiseConv2DBatchNormReLU6Fold."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6}, [
        LayerPattern('BatchNormalization', {},
                     [LayerPattern('DepthwiseConv2D', {}, [])])
    ])

  def _map_to_conv_bn_layer(self, conv_layer, bn_layer, relu_layer=None):
    conv_bn_params = _get_conv_bn_params(conv_layer, bn_layer, relu_layer)
    return _DepthwiseConvBatchNorm2D(**conv_bn_params)

  def custom_objects(self):
    return {
        '_DepthwiseConvBatchNorm2D': _DepthwiseConvBatchNorm2D
    }
