import numpy as np
import tensorflow as tf
from copy import copy

def gen_to_scale(gen, pmax, pmin):
    return np.divide(np.subtract(gen,pmin.T),(pmax-pmin).T)

def scale_to_gen(scale, pmax, pmin):
    return tf.math.multiply(scale,(pmax-pmin).reshape(len(pmin)))+pmin.reshape(len(pmin))

def load_to_scale(load, base_load, x):
    return (np.divide(load,base_load.T)-1+x)/(2*x)

def scale_to_load(scale, base_load, x):
    return base_load*(scale*2*x+1-x)

def get_slack_bus_gen(gen, load):
    slack = (tf.reduce_sum(load, axis=1, keepdims=True)-tf.reduce_sum(gen, axis=1, keepdims=True))
    return tf.concat([slack, gen], axis=1)
def get_slack_bus_gen2(gen, load):
    slack = (tf.reduce_sum(load, axis=1, keepdims=True)-tf.reduce_sum(gen, axis=1, keepdims=True))
    return slack

def load_to_input(load, base_load, x):
    assert load.shape[1]==len(base_load)
    return load_to_scale(load, base_load, x)

'''def input_to_load(input, base_load, x):
    active_base_load = base_load[np.nonzero(base_load)]
    assert input.shape[1]==active_base_load.shape[0]
    load = scale_to_load(input, active_base_load, x)
    for i in np.where(base_load==0)[0]:
        load = np.insert(load, i, np.zeros(len(load)), axis=1)
    return load'''

def input_to_load(input, base_load, x):
    active_base_load = base_load[np.nonzero(base_load)]
    assert input.shape[1]==active_base_load.shape[0]
    load = scale_to_load(input, active_base_load, x)
    for i in np.where(base_load==0)[0]:
        load = np.insert(load, i, tf.zeros(load.shape[0]), axis=1)
    return load

def gen_to_output(gen, pmax, pmin):
    active_gen = np.delete(gen, np.argwhere(np.all(gen[..., :] == 0, axis=0)), axis=1)
    active_pmax = np.delete(pmax, np.argwhere(np.all(gen[..., :] == 0, axis=0)))
    active_pmin = np.delete(pmin, np.argwhere(np.all(gen[..., :] == 0, axis=0)))
    return gen_to_scale(active_gen, active_pmax, active_pmin)

'''def output_to_gen(output, pmax, pmin):
    full_output = fill_gen(output, pmax)
    return scale_to_gen(full_output, pmax, pmin)'''

def output_to_gen(output, pmax_mat):
    return tf.linalg.matmul(output, pmax_mat)

'''def get_angles(gen, load, Bmat):
    return np.hstack((np.zeros((len(gen),1)),np.matmul((gen-load)[:,1:], np.linalg.inv(Bmat[1:,1:]))))'''

def get_angles(gen, load, Bmat):
    Binv=tf.linalg.inv(Bmat[1:,1:])
    diff = (gen-load)[:,1:]
    return tf.matmul(diff, Binv)

def fill_gen(gen, pmax):
    full_gen = np.zeros((gen.shape[0], pmax.shape[0]))
    idx=(pmax!=0)
    idx[0]=False #Keep slack bus at zero
    full_gen[:,idx]=gen.reshape(gen.shape[0],gen.shape[1])
    return full_gen

def calculate_cost(gen, cost):
    
    c1=cost[:,0].reshape(len(cost),1)
    c0=cost[:,1].reshape(len(cost),1)
    gen2 = tf.square(gen)

    total_cost=tf.matmul(gen2, c1)+tf.matmul(gen, c0)
    return tf.squeeze(total_cost)

def find_factors(x):
    s = np.ceil(np.sqrt(x))
    for t in range(int(s))[::-1]:
        if x%t==0:
            return int(min(t, x/t)), int(max(t, x/t))