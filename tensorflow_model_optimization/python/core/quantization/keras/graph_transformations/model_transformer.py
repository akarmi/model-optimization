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
# pylint: disable=g-explicit-length-test
"""Apply graph transformations to a Keras model."""

import collections

from tensorflow.python import keras

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms as transforms_mod

LayerNode = transforms_mod.LayerNode


class ModelTransformer(object):
  """Matches patterns to apply transforms in a Keras model graph."""

  def __init__(self, model, transforms, candidate_layers=None):
    if not (isinstance(model, keras.Model) and model._is_graph_network):  # pylint: disable=protected-access
      raise ValueError('Only Keras functional models can be transformed.')

    self.model = model
    self.transforms = transforms
    self.candidate_layers = candidate_layers

  def _get_consuming_layers(self, check_layer):
    """Returns all the layers which are out nodes from the layer."""
    consuming_layers = []
    for layer in self._config['layers']:
      for inbound_node in layer['inbound_nodes']:
        for connection_info in inbound_node:
          if connection_info[0] == check_layer['name']:
            consuming_layers.append(layer)
    return consuming_layers

  def _get_output_consumers(self, check_layer):
    """Returns if any tensors from the layer are outputs of the model."""
    output_consumers = []
    for output_layer in self._config['output_layers']:
      if output_layer[0] == check_layer['name']:
        output_consumers.append(output_layer)
    return output_consumers

  def _get_layers(self, layer_names):
    return [
        layer for layer in self._config['layers']
        if layer['name'] in layer_names
    ]

  def _get_layer_weights(self, layer_name):
    return self._layer_weights_map.get(layer_name, {})

  def _has_wrapper(self, layer):
    return 'layer' in layer['config']

  def _get_unwrapped_layer(self, layer):
    return layer['config']['layer']

  def _match_layer(self, layer, pattern):
    """Check if specific layer matches the pattern."""

    if self.candidate_layers and layer['name'] not in self.candidate_layers:
      return False

    # TODO(pulkitb): Possible changes and extensions to this method.
    # Consider making this case insensitive
    # Add support for multiple types. (Conv2D|DepthwiseConv2d)
    #
    # TODO(tfmot): has_wrapper() check can be removed after using
    # candidate_layers parameter in callers.
    if not self._has_wrapper(layer) or self._get_unwrapped_layer(
        layer)['class_name'] != pattern.class_name:
      return False
    layer_config = self._get_unwrapped_layer(layer)['config']
    for key, value in pattern.config.items():
      # This comparison should probably use the serialized value.
      # Consider adding regex support to key/values as well. This will allow
      # negative matches as well.
      if layer_config.get(key) != value:
        return False

    return True

  def _match_layer_with_inputs(self, layer, pattern):
    """Match pattern at this layer, and continue to match at its inputs."""

    if not self._match_layer(layer, pattern):
      return None

    inbound_nodes = layer['inbound_nodes']
    if len(inbound_nodes) > 1:
      # `layer` is re-used for more than 1 connection from previous layers. If
      # a pattern matches one set of inputs and is replaced, it will break the
      # other connection.
      #
      # Note that theoretically it's possible to have multiple connections have
      # exactly the same pattern, and in that case the transform might be
      # applied. But that's a very complicated edge case not worth handling.
      return None

    # If a layer has multiple inbound nodes, it will produce multiple outbound
    # connections as well. Hence no need to explicitly check that.

    consuming_layers = self._get_consuming_layers(layer)
    output_consumers = self._get_output_consumers(layer)
    if len(consuming_layers) + len(output_consumers) > 1:
      # Even if a layer has only 1 incoming connection, multiple layers may
      # still consume the output. This is problematic since transforming this
      # layer can leave the other consuming nodes not part of the pattern
      # dangling.
      #
      # Note that having multiple consumers is only a problem for intermediate
      # layers, and not the head node of the sub-tree.
      # TODO(pulkitb): Consider support for this exemption for tip of the tree.
      return None

    if len(pattern.inputs) == 0:
      # Leaf layer in pattern.
      return LayerNode(layer, self._get_layer_weights(layer['name']), [])

    # There is a possible edge case where a single layer may output multiple
    # tensors and multiple tensors from that layer may be used by the
    # connection. Ignoring those for now.
    input_layer_names = [
        connection_info[0] for connection_info in inbound_nodes[0]]
    input_layers = self._get_layers(input_layer_names)

    if len(input_layers) != len(pattern.inputs):
      # Number of inputs this layer takes is different from the number of
      # inputs in the pattern.
      #
      # This path currently has the limitation that it requires an exact number
      # of inputs to match a pattern. For example, if a user wants to match
      # 2 Convs -> Concat and 3 Convs -> Concat, they would need to write
      # 2 different patterns.
      return None

    # Inbound layers can have different order from the list of input patterns.
    # TODO(pulkitb): Fix by checking all permutations.
    input_match_layer_nodes = []
    for input_layer, pattern in zip(input_layers, pattern.inputs):
      match_layer_node = self._match_layer_with_inputs(input_layer, pattern)
      if not match_layer_node:
        return None
      input_match_layer_nodes.append(match_layer_node)

    return LayerNode(
        layer, self._get_layer_weights(layer['name']), input_match_layer_nodes)

  def _find_pattern(self, pattern):
    for layer in self._config['layers']:
      match_layer = self._match_layer_with_inputs(layer, pattern)
      if match_layer:
        return match_layer

    return None

  def _get_leaf_layers(self, match_layer):
    """Return leaf layers from this sub-graph tree."""

    if not match_layer.input_layers:
      return [match_layer.layer]

    # If 2 different layers point to the same input, or if a layer uses the
    # same input multiple times, the input layer can be repeated. But it
    # preserves a bit of structure.

    leaf_layers = []
    for inp in match_layer.input_layers:
      leaf_layers.extend(self._get_leaf_layers(inp))

    return leaf_layers

  def _replace(self, match_layer_node, replacement_layer_node):
    """Replace the tree of match_layer_node with replacement_layer_node."""

    # 1. Point all consumers of the head of the matching sub-tree to the head
    # replacement layer.
    #
    # There are some assumptions baked in. The head layer only has 1 inbound and
    # outbound node. The resulting number and shape of tensors from the
    # replaced layer should equal the original layer.

    consuming_layers = self._get_consuming_layers(match_layer_node.layer)
    for consumer in consuming_layers:
      for inbound_node in consumer['inbound_nodes']:
        for connection_info in inbound_node:
          if connection_info[0] == match_layer_node.layer['name']:
            connection_info[0] = replacement_layer_node.layer['name']

    output_consumers = self._get_output_consumers(match_layer_node.layer)
    for output_consumer in output_consumers:
      output_consumer[0] = replacement_layer_node.layer['name']

    # 2. Create inbound nodes for the replacement layers. This connects all
    # the replacement layers.

    def _assign_inbounds_for_replacement(layer_node):
      """_assign_inbounds_for_replacement."""

      if not layer_node.input_layers:
        return

      layer_node['inbound_nodes'] = [[]]
      for input_layer in layer_node.input_layers:
        # inbound_nodes can be specific tensors from multiple inbound
        # connections. We make the following assumptions.
        # - Only 1 inbound node for each replacement layer.
        # - Only 1 tensor output from the previous layer which is connected.
        # - call() method during construction does not have any args.
        # These are reasonable assumptions for almost all case we are
        # interested in.
        layer_node['inbound_nodes'][0].append(
            [input_layer.layer['name'], 0, 0, {}])

        _assign_inbounds_for_replacement(input_layer)

    _assign_inbounds_for_replacement(replacement_layer_node)

    # 3. Connect the leaves of the replacement_layers to the inbound_nodes of
    # the leaves in the original layer.

    original_leaf_layers = self._get_leaf_layers(match_layer_node)
    original_inbound_nodes = [
        layer['inbound_nodes'] for layer in original_leaf_layers]

    replacement_leaf_layers = self._get_leaf_layers(replacement_layer_node)

    # The original pattern and the replacement pattern can potentially have
    # different number of leaf nodes and differences in how they consume these
    # input layers. Matching them will require sophisticated hackery to recreate
    # the new layers with the original input structure.

    # Given our existing transforms, we can assume they match.

    if len(original_leaf_layers) != len(replacement_leaf_layers):
      raise RuntimeError('Different size of leaf layers not supported yet.')

    for original_inbound_nodes, replacement_leaf_layer in zip(
        original_inbound_nodes, replacement_leaf_layers):
      replacement_leaf_layer['inbound_nodes'] = original_inbound_nodes

    # 4. Remove the original matched layers

    def _get_layer_names(layer_node):
      result = [layer_node.layer['name']]
      for input_layer in layer_node.input_layers:
        result.extend(_get_layer_names(input_layer))
      return result

    layers_to_remove_names = _get_layer_names(match_layer_node)
    layers_to_remove = self._get_layers(layers_to_remove_names)

    for layer_to_remove in layers_to_remove:
      self._config['layers'].remove(layer_to_remove)
    # Remove entry from weight map, now that layer has been removed.
    for layer_name in layers_to_remove_names:
      self._layer_weights_map.pop(layer_name, None)

    # 5. Add in the new layers.

    def _add_replacement_layer(layer_node):
      self._config['layers'].append(layer_node.layer)
      if layer_node.weights:
        self._layer_weights_map[layer_node.layer['name']] = layer_node.weights

      for input_layer in layer_node.input_layers:
        _add_replacement_layer(input_layer)

    _add_replacement_layer(replacement_layer_node)

  def _get_keras_layer_weights(self, keras_layer):
    """Returns a map of weight name, weight matrix. Keeps keras ordering."""
    weights_map = collections.OrderedDict()
    for weight_tensor, weight_numpy in \
        zip(keras_layer.weights, keras_layer.get_weights()):
      weights_map[weight_tensor.name] = weight_numpy

    return weights_map

  def transform(self):
    """Transforms the Keras model by applying all the specified transforms.

    This is the main entry point function used to apply the transformations to
    the Keras model.

    This requires that each layer in the Keras model is wrapped by a Keras
    wrapper, which may be used to propagate any data to the transforms that is
    usecase specific. The wrapper must provide access to the layer via
    self.layer.
    TODO(tfmot): formalize this contract.

    Not suitable for multi-threaded use. Creates and manipulates internal state.

    Returns:
      Keras model after transformation.
    """

    # Gets a serialized dict representation of the model, containing all its
    # layers, their connections and configuration. This is the main structure
    # which is used to understand model structure, and also manipulate it.
    #
    # config = {
    #   'input_layers': [ ... ],
    #   'layers': [{
    #       'inbound_nodes': [INPUT CONFIG OF LAYER],
    #       'name': 'LAYER_NAME',
    #       'config': { LAYER_CONFIG }
    #     }, {
    #     ...
    #   }],
    #   'output_layers': [ ... ],
    #   'name': 'MODEL_NAME',
    #
    self._config = self.model.get_config()

    self._layer_weights_map = {}
    for layer in self.model.layers:
      self._layer_weights_map[layer.name] = self._get_keras_layer_weights(layer)

    # We run an infinite loop and keep applying transformations as long as
    # patterns are found. This allows recursive pattern matching where a
    # modification by one transform may lead to another match.
    #
    # TODO(pulkitb): This leads to infinite loops with poor patterns which may
    # match their replacement. Add counters with limits to fix it.
    while True:
      match_found = False
      for transform in self.transforms:
        # A transform may find multiple instances of a pattern in the model.
        # Keep finding and replacing till done.
        while True:
          match_layer_node = self._find_pattern(transform.pattern())
          if not match_layer_node:
            break

          replacement_layer_node = transform.replacement(match_layer_node)

          # If equal, the matched layers are being replaced with exactly the
          # same set of layers that were matched with the same config.
          # For Transforms, that may inadvertently do this we can end up in
          # an infinite loop. Break if no meaningful change has been made.
          if match_layer_node == replacement_layer_node:
            break

          match_found = True
          self._replace(match_layer_node, replacement_layer_node)

      # None of the transforms found a pattern. We can stop now.
      if not match_found:
        break

    custom_objects = {}
    for transform in self.transforms:
      custom_objects.update(transform.custom_objects())

    # Reconstruct model from the config, using the cloned layers.
    transformed_model = keras.Model.from_config(self._config, custom_objects)

    for layer in transformed_model.layers:
      weights = self._layer_weights_map.get(layer.name)
      if weights:
        layer.set_weights(list(weights.values()))

    return transformed_model
