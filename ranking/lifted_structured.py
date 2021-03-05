import tensorflow as tf
from ranking import common
from tensorflow.python.ops import math_ops

def lifted_loss(labels,embeddings,margin):
    """Computes the lifted structured loss.
        Args:
          labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
            multiclass integer labels.
          embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
          margin: Float, margin term in the loss definition.
        Returns:
          lifted_loss: tf.float32 scalar.
        """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = tf.shape(labels)
    assert lshape.shape == 1
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = common.pairwise_distance(embeddings,squared=True)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    diff = margin - pairwise_distances
    mask = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = tf.math.reduce_min(diff, 1, keepdims=True)
    row_negative_maximums = tf.math.reduce_max(
        tf.math.multiply(diff - row_minimums, mask), 1,
        keepdims=True) + row_minimums

    # Compute the loss.
    # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
    #   where m_i is the max of alpha - negative D_i's.
    # This matches the Caffe loss layer implementation at:
    #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp  # pylint: disable=line-too-long

    max_elements = tf.math.maximum(row_negative_maximums,
                                   tf.transpose(row_negative_maximums))
    diff_tiled = tf.tile(diff, [batch_size, 1])
    mask_tiled = tf.tile(mask, [batch_size, 1])
    max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

    loss_exp_left = tf.reshape(
        tf.math.reduce_sum(
            tf.math.multiply(
                tf.math.exp(diff_tiled - max_elements_vect), mask_tiled),
            1,
            keepdims=True), [batch_size, batch_size])

    loss_mat = max_elements + tf.math.log(loss_exp_left +
                                          tf.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = tf.cast(
        adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size]))

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = tf.math.reduce_sum(mask_positives) / 2.0

    lifted_loss = tf.math.truediv(
        0.25 * tf.math.reduce_sum(
            tf.math.square(
                tf.math.maximum(
                    tf.math.multiply(loss_mat, mask_positives), 0.0))),
        num_positives)
    return lifted_loss
