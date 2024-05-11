import numpy as np
import tensorflow as tf

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)