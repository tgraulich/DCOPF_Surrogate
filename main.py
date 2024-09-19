import argparse
import os
import sys
import time

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_dir)
os.chdir(project_dir)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mat4py import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from processing import *
import tensorflow as tf
import tensorflow.keras.backend as K

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-k', '--key', default=0, type=int, help='Random Key to use')#Replace with random number
    argparser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='The learning rate')
    argparser.add_argument('-e', '--epochs', default=200, type=int, help='The number of epochs to run')
    argparser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size to use')
    argparser.add_argument('-l2', '--l2_scale', default=1e-3, type=int, help='Scale for L2 Regularization')
    argparser.add_argument('-r', '--rate', default=0.0, type=float, help='Dropout Rate') #Maybe implement
    argparser.add_argument('--use_schedule', dest='schedule', action='store_true', help='Whether to use a schedule for the learning rate') #Maybe implement
    argparser.add_argument('--data_dir', default='sample_data30.mat', help='Directory to take input data from')
    argparser.add_argument('--res_dir', default='results/res_', help='Results directory')
    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('--baseline', dest='baseline', action='store_true', help='Use a combined loss function for boosted training.')
    argparser.add_argument('-vw', '--violation_weight', default=0, type=float, help="Weight with which constraint violation get added to the loss function")
    argparser.set_defaults(debug_load=False, baseline=True, schedule=False)

    args = argparser.parse_args()
 
    return args

def calculate_violation(divider, pmax_mat, Bmat, Amat):
    def _calculate_violation(ytrue, ypred):
        inputs=ytrue[:,divider:]
        ytrue = ytrue[:,:divider]
        ypred = ypred[:,:divider]
        ogen=output_to_gen(ypred, pmax_mat)
        angles=get_angles(ogen, inputs, Bmat)
        full_angles=tf.concat([tf.zeros((tf.shape(angles)[0],1), dtype=tf.dtypes.float64), angles], axis=1)
        return K.max([-1e-3,K.max(tf.square(tf.matmul(Amat, tf.transpose(full_angles)))-1)])
    return _calculate_violation

def baseline_loss(divider):
    def _baseline_loss(ytrue, ypred):
        ytrue = ytrue[:,:divider]
        ypred = ypred[:,:divider]
        return K.mean(tf.abs(ytrue - ypred))
    return _baseline_loss

def combined_loss(divider, pmax_mat, Bmat, Amat, violation_weight):
    def _combined_loss(ytrue, ypred):
        return baseline_loss(divider)(ytrue, ypred)+violation_weight*calculate_violation(divider, pmax_mat, Bmat, Amat)(ytrue, ypred)
    return _combined_loss
def main():

    start_time = time.time()
    config = parse_args()

    learning_rate   = config.learning_rate
    num_epochs      = config.epochs
    batch_size      = config.batch_size
    l2_scale        = config.l2_scale
    violation_weight = config.violation_weight
    if config.debug_load:
        print("Running in Debug mode (2 epochs)...")
        num_epochs = 2

    print('Hyperparameters:\n Learning Rate: {}\n Number of Epochs: {}\n Batch Size: {}\n L2 Scale: {}\n'.format(learning_rate, num_epochs, batch_size, l2_scale))

    print("Loading Data...")
    i_dir = config.data_dir

    data = loadmat(i_dir)

    load=np.array(data["load_samples_full"])
    gen=np.array(data["generator_samples"])
    divider=gen.shape[1]-1
    gen_bus=np.array(data["generator_buses"])
    gen_full=np.zeros(load.shape)
    gen_full[:,gen_bus-1]=gen.reshape(gen.shape[0],gen.shape[1],1)
    gen=gen_full
    pmin=np.array(data["pmin"])
    pmax=np.array(data["pmax"])
    pmin_full=np.zeros(load.shape[1])
    pmin_full[gen_bus-1]=pmin
    pmin=pmin_full
    pmax_full=np.zeros(load.shape[1])
    pmax_full[gen_bus-1]=pmax
    #pmax=pmax_full
    gen_bus_mat=np.zeros((len(pmax), len(pmax_full)))
    for i in range(len(gen_bus)):
        gen_bus_mat[i,gen_bus[i]-1]=1
    pmax_mat=np.matmul(np.diag(pmax[:,0]), gen_bus_mat)[1:,:]
    base_load=np.array(data["load"])
    #outputs=gen_to_output(gen_full, pmax_full, pmin_full)[:,1:]

    Bmat=np.array(data["Bbus"])
    Amat=np.array(data["Amat"])
    voltage_angles=np.array(data["voltages_angles"])*np.pi/1.8
    voltage_angles=np.subtract(voltage_angles,voltage_angles[:,0].T.reshape(len(voltage_angles),1))
    x= data["sampling_range"]

    print("Preparing data for training...")

    i_train, i_test, o_train, o_test = train_test_split(load, gen_full, test_size=0.1, random_state=42, shuffle=False)

    scales_train = gen_to_output(o_train, pmax_full, pmin_full)[:,1:]
    scales_test = gen_to_output(o_test, pmax_full, pmin_full)[:,1:]

    output_train=np.append(scales_train, i_train, axis=1)
    output_test=np.append(scales_test, i_test, axis=1)

    input_train = load_to_input(i_train, base_load, x)
    input_test = load_to_input(i_test, base_load, x)

    print("Starting training...")

    tf.keras.backend.set_floatx('float64')
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_train.shape[1],)))
    model.add(tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=output_train.shape[1], activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=combined_loss(divider, pmax_mat, Bmat, Amat, violation_weight=violation_weight), optimizer=opt, metrics=[baseline_loss(divider), calculate_violation(divider, pmax_mat, Bmat, Amat)])
    hist = model.fit(input_train, output_train, verbose=1, epochs=num_epochs, validation_split=0.05)

if __name__ == '__main__':
    main()