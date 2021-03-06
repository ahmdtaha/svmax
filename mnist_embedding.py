import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
from ranking import contrastive
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist


class ToyModel(Model):
    def __init__(self, emb_dim):
        super(ToyModel, self).__init__()
        self.emb_dim = emb_dim
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(self.emb_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = tf.nn.l2_normalize(x, -1)
        return x
    
def sample_k_imgs_for_pid(pid, all_imgs, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    possible_fids = tf.boolean_mask(all_imgs, tf.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original FID count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.math.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
    full_range = tf.math.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random.shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

    return selected_fids, tf.fill([batch_k], pid)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    svmax_enabled = True
    num_instance = 4
    num_cls = 8
    batch_size = num_cls * num_instance

    mnist = tf.keras.datasets.mnist

    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("Num GPUs:", len(physical_devices))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    
    unique_pids = np.unique(y_train)
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))
    dataset = dataset.take((len(unique_pids) // num_cls) * num_cls)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

    dataset = dataset.map(lambda pid: sample_k_imgs_for_pid(
        pid, all_imgs=x_train, all_pids=y_train, batch_k=num_instance))

    # Ungroup/flatten the batches for easy loading of the files.
    dataset = dataset.unbatch()
    train_ds = dataset.batch(batch_size)

    
    test_ds_size = batch_size*10
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).take(test_ds_size).batch(batch_size)

    

    # Create an instance of the model
    model = ToyModel(2)
    loss_object = contrastive.contrastive_loss
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    def largest_singular_upperbound(batch_size):
        return np.sqrt(batch_size)

    def nuc_norm_upperbound(batch_size, dim):
        return np.sqrt((batch_size * dim) / np.maximum(batch_size, dim)) * np.sqrt(batch_size)

    def svd_mean(param_norm_batch_embedding_tiled, param_batch_size):
        sing_values = tf.linalg.svd(
            tf.reshape(param_norm_batch_embedding_tiled, [param_batch_size, model.emb_dim]),  # num_tiles * batch_size
            compute_uv=False)
        mean_sing_val = tf.reduce_mean(sing_values)
        return mean_sing_val

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            model_output = model(images, training=True)
            contrastive_idx = np.tile([0, 1, 4, 3, 2, 5, 6, 7], num_cls // 2)
            for i in range(num_cls // 2):
                contrastive_idx[i * 8:i * 8 + 8] += i * 8

            contrastive_idx = np.expand_dims(contrastive_idx, 1)
            batch_embedding_ordered = tf.gather_nd(model_output, contrastive_idx)
            pids_ordered = tf.gather_nd(labels, contrastive_idx)
            embeddings_anchor, embeddings_positive = tf.unstack(
                tf.reshape(batch_embedding_ordered, [-1, 2, model.emb_dim]), 2,
                1)
    
            fixed_labels = np.tile([1, 0, 0, 1], num_cls // 2)
            labels = tf.constant(fixed_labels)
            mean_sing_val = svd_mean(batch_embedding_ordered,  batch_size)
            lower_bound = largest_singular_upperbound( batch_size) / model.emb_dim
            upper_bound = nuc_norm_upperbound( batch_size, model.emb_dim) / model.emb_dim

            if svmax_enabled:
                reg_loss = 1 * tf.exp((upper_bound - mean_sing_val) / (upper_bound - lower_bound))
            else:
                reg_loss = 0


            loss = loss_object(labels, embeddings_anchor, embeddings_positive) + reg_loss

            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        # train_accuracy(labels, predictions)

    
    EPOCHS = 200

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
       
        cnt = 0
        for images, labels in train_ds:
            train_step(images, labels)
            cnt +=1
            if cnt == len(y_train) // batch_size:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1,train_loss.result()))
                num_test_batches = 0
                test_embedding = np.zeros((test_ds_size,2))
                classes_colors = ['green', 'yellow', 'orange', 'purple', 'red', 'black', 'hotpink', 'cyan', 'navy',
                                  'indigo', 'goldenrod', 'maroon', 'grey', 'olive']
                display_clrs = []
                for test_images, test_labels in test_ds:
                    test_embedding[num_test_batches*batch_size:((num_test_batches+1)*batch_size),:] = model(test_images, training=False)
                    display_clrs.extend([classes_colors[lbl] for lbl in test_labels])
                    num_test_batches+=1

                # print(num_test_batches)


                plt.scatter(test_embedding[:, 0], test_embedding[:, 1],c=display_clrs, marker='o', s=20.0)
                plt.show()
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
                plt.savefig('./svmax_{}_{:03d}.png'.format(svmax_enabled,epoch))
                plt.close()
                break
            

if __name__ == '__main__':
    main()
