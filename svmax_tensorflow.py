import numpy as np
import tensorflow as tf


def nuc_norm_upperbound(batch_size, dim):
    return np.sqrt((batch_size * dim) / np.maximum(batch_size, dim)) * np.sqrt(batch_size)

def svmax_lower_bound(batch_size,dim):
    return np.sqrt(batch_size) / dim

def svmax_upper_bound(batch_size, dim):
    return nuc_norm_upperbound(batch_size, dim) / dim

def svd_mean(mini_batch_embedding, batch_size, dim):

    assert batch_size > dim , 'batch size should be bigger than emb_dim.' \
                              'This is not required, but be aware of your bounds if you break this constraint'
    sing_values = tf.linalg.svd(
        tf.reshape(mini_batch_embedding, [batch_size, dim]),
        compute_uv=False)
    mean_sing_val = tf.reduce_mean(sing_values)

    return mean_sing_val

def main():
    batch_size = 144
    emb_dim = 128
    rand_emb = tf.random.uniform([batch_size,emb_dim])
    normalized_emb = tf.nn.l2_normalize(rand_emb, -1)
    sing_mu = svd_mean(normalized_emb,batch_size,emb_dim).numpy()

    upper_bound = svmax_upper_bound(batch_size,emb_dim)
    lower_bound = svmax_lower_bound(batch_size,emb_dim)
    print('L: {:.4f} Mu: {:.4f} U: {:.4f}' .format(lower_bound,sing_mu,upper_bound))

    assert sing_mu <= upper_bound and sing_mu >= lower_bound, 'Revise your bounds'



    lower_case_embedding = np.zeros((batch_size,emb_dim))
    lower_case_embedding [:,1]= 1
    normalized_emb = tf.convert_to_tensor(lower_case_embedding) ## already ||.||_2= 1
    sing_mu = svd_mean(normalized_emb, batch_size, emb_dim).numpy()
    print('L: {:.4f} Mu: {:.4f} U: {:.4f}'.format(lower_bound, sing_mu, upper_bound))
    assert sing_mu <= upper_bound and np.isclose(sing_mu , lower_bound), 'Revise your bounds'

if __name__ == '__main__':
    main()