import tensorflow as tf
from ranking import common

def masked_minimum_idx(data, mask, dim=1):
    delta = 1
    ## The delta provides a cheap workaround to avoid a wrong argmin result when a single element is possible
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)+delta

    masked_minimums_idx = tf.math.argmin(
        tf.math.multiply(data - axis_maximums, mask), dim) ## Not the max  element is definitely less than 0
    return masked_minimums_idx
def masked_maximum_idx(data, mask, dim=1):
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)

    masked_maximums_idx = tf.math.argmax(
        tf.math.multiply(data - axis_minimums, mask), dim)
    return masked_maximums_idx

def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
  masked_minimums = tf.math.reduce_min(
      tf.math.multiply(data - axis_maximums, mask), dim,
      keepdims=True) + axis_maximums
  return masked_minimums



def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
  masked_maximums = tf.math.reduce_max(
      tf.math.multiply(data - axis_minimums, mask), dim,
      keepdims=True) + axis_minimums
  return masked_maximums

def triplet_semihard_loss_apn(embeddings,labels):
    lshape = tf.shape(labels)
    assert lshape.shape == 1
    labels = tf.reshape(labels, [lshape[0], 1])
    batch_size = tf.size(labels)
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    adjacency_not = tf.math.logical_not(adjacency)

    # Build pairwise squared distance matrix.
    pdist_matrix = common.pairwise_distance(embeddings, squared=True)

    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            pdist_matrix_tile, tf.reshape(
                tf.transpose(pdist_matrix), [-1, 1])))

    mask_final = tf.reshape(
        tf.math.greater(
            tf.reduce_sum(
                tf.cast(mask, dtype=tf.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)


    adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    negatives_outside_idx = tf.reshape(masked_minimum_idx(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside_idx = tf.transpose(negatives_outside_idx)


    negatives_inside_idx = tf.tile(masked_maximum_idx(pdist_matrix, adjacency_not)[:, tf.newaxis],
                                          [1, batch_size])

    semi_hard_negatives_idx = tf.where(
        mask_final, negatives_outside_idx, negatives_inside_idx)

    mask_positives = tf.cast(
        adjacency, dtype=tf.float32) - tf.linalg.diag(
        tf.ones([batch_size]))

    #print(tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, batch_size]).numpy)
    range_mask = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, batch_size])
    anchor_idx = tf.boolean_mask(range_mask,
                                 tf.cast(mask_positives, tf.bool))

    #print(tf.tile(tf.reshape(tf.range(0, batch_size), [1, batch_size]), [batch_size, 1]))
    positive_idx = tf.boolean_mask(tf.transpose(range_mask),
                                   tf.cast(mask_positives, tf.bool))

    negative_idx = tf.boolean_mask(semi_hard_negatives_idx, tf.cast(mask_positives, tf.bool))

    return anchor_idx,positive_idx,negative_idx

def triplet_hard_loss_apn(embeddings,labels):
    lshape = tf.shape(labels)
    assert lshape.shape == 1
    labels = tf.reshape(labels, [lshape[0], 1])
    batch_size = tf.size(labels)
    delta = 0.5

    same_identity_mask = tf.squeeze(tf.equal(tf.expand_dims(labels, axis=1),
                                  tf.expand_dims(labels, axis=0)))
    # same_identity_mask = tf.squeeze(same_identity_mask)
    negative_mask = tf.math.logical_not(same_identity_mask)
    positive_mask = tf.math.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(labels)[0], dtype=tf.bool))

    anchor_idx = tf.range(0, batch_size)
    dists = common.pairwise_distance(embeddings, squared=True)
    positive_idx = tf.argmax(tf.multiply((dists + delta), tf.cast(positive_mask, tf.float32)), axis=1)
    axis_maximums = tf.reduce_max(dists) + delta
    negative_idx = tf.cast(tf.math.argmin(tf.math.multiply(dists - axis_maximums, tf.cast(negative_mask, tf.float32)),
                                   axis=1),tf.int32)  ## Not the max  element is definitely less than 0

    return anchor_idx,positive_idx,negative_idx