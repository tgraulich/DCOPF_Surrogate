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
import matplotlib.pyplot as plt

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
    argparser.add_argument('-l2', '--l2_scale', default=1e-3, type=float, help='Scale for L2 Regularization')
    argparser.add_argument('-r', '--rate', default=0.0, type=float, help='Dropout Rate') #Maybe implement
    argparser.add_argument('--use_schedule', dest='schedule', action='store_true', help='Whether to use a schedule for the learning rate') #Maybe implement
    argparser.add_argument('--data_dir', default='sample_data30.mat', help='Directory to take input data from')
    argparser.add_argument('--res_dir', default='results/res_', help='Results directory')
    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('-vw', '--violation_weight', default=0, type=float, help="Weight with which constraint violation get added to the loss function")
    argparser.add_argument('-eps', '--epsilon', default=1e-3, type=float, help="Bound below which negative values of violation contribute to loss function")
    argparser.set_defaults(debug_load=False, baseline=True, schedule=False)

    args = argparser.parse_args()
 
    return args

def baseline_loss(divider):
    def _baseline_loss(ytrue, ypred):
        ytrue = ytrue[:,:divider]
        ypred = ypred[:,:divider]
        return K.mean(tf.divide(tf.abs(ytrue - ypred),ytrue))
    return _baseline_loss

def cost_metric(cost, divider, pmax, pmin):
    def _cost_metric(ytrue, ypred):
        true_scale = ytrue[:,:divider]
        pred_scale = ypred[:,:divider]
        load = ytrue[:,divider:]
        true_gen = scale_to_gen(true_scale, pmax[1:], pmin[1:])
        pred_gen = scale_to_gen(pred_scale, pmax[1:], pmin[1:])
        true_gen = get_slack_bus_gen(true_gen, load)
        pred_gen = get_slack_bus_gen(pred_gen, load)
        true_cost= calculate_cost(true_gen, cost)
        pred_cost = calculate_cost(pred_gen, cost)
        return K.mean(pred_cost-true_cost)
    return _cost_metric

def main():

    start_time = time.time()
    config = parse_args()

    learning_rate    = config.learning_rate
    num_epochs       = config.epochs
    batch_size       = config.batch_size
    l2_scale         = config.l2_scale
    violation_weight = config.violation_weight
    eps              = config.epsilon
    res_dir = config.res_dir
    if config.debug_load:
        print("Running in Debug mode (2 epochs)...")
        num_epochs = 2
        res_dir="results/res_"



    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    f = open('{}/summary.txt'.format(config.res_dir), 'w')
    f.write('##### Summary: #####\n\n')
    f.write('### Hyperparameters: ###\n')
    f.writelines('\n'.join(['Random Key: {}'.format(config.key), 'Learning Rate: {}'.format(learning_rate), 'Batch Size: {}'.format(batch_size), 'L2 Scale: {}'.format(l2_scale), 'Violation Weight: {}'.format(violation_weight)]))
    f.close()
    print('Hyperparameters:\n Learning Rate: {}\n Number of Epochs: {}\n Batch Size: {}\n L2 Scale: {}\n'.format(learning_rate, num_epochs, batch_size, l2_scale))

    print("Loading Data...")
    i_dir = config.data_dir

    data = loadmat(i_dir)

    load=np.array(data["load_samples_full"])
    cost=np.array(data["cost"])
    gen=np.array(data["generator_samples"])
    divider = gen.shape[1]-1
    pmin=np.array(data["pmin"])
    pmax=np.array(data["pmax"])
    base_load=np.array(data["load"])

    x= data["sampling_range"]

    fig, axs = plt.subplots(2,3, figsize=(10,7))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.hist(gen[:,i], bins=20, histtype="step", label="Pred", linestyle="--", color="red")

    fig.savefig("{}/generators.png".format(res_dir))

    print("Preparing data for training...")

    i_train, i_test, o_train, o_test = train_test_split(load, gen, test_size=0.1, random_state=config.key)

    train_cost = calculate_cost(o_train, cost)
    test_cost = calculate_cost(o_test, cost)

    scales_train = gen_to_output(o_train, pmax, pmin)[:,1:]
    scales_test = gen_to_output(o_test, pmax, pmin)[:,1:]

    output_train=np.append(scales_train, i_train, axis=1)
    output_test=np.append(scales_test, i_test, axis=1)

    input_train = load_to_input(i_train, base_load, x)
    input_test = load_to_input(i_test, base_load, x)
    
    print("Input shape: {}, Output shape: {}".format(input_train.shape, output_train.shape))

    print("Starting training...")

    tf.keras.backend.set_floatx('float64')
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_train.shape[1],)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=output_train.shape[1], activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=baseline_loss(divider), optimizer=opt, metrics=[cost_metric(cost, divider, pmax, pmin)])
    hist = model.fit(input_train, output_train, verbose=1, epochs=num_epochs, validation_split=0.05, batch_size=batch_size)

    print("Evaluating model...")

    predictions = model.predict(input_test)[:,:divider]
    pred_gen = np.array(scale_to_gen(predictions, pmax[1:], pmin[1:]))
    pred_gen = get_slack_bus_gen(pred_gen, i_test)
    pred_cost=calculate_cost(pred_gen, cost)

    #print(true_cost-test_cost)
    #print(cost)
    #num_violations = count_violation(pred_gen, i_test, pmax_mat, Bmat, Amat)
    #print("{}/{} Test cases were infeasible".format(num_violations, len(i_test)))

    test_loss = model.evaluate(input_test, output_test)

    print("Saving results")

    np.save('{}/train_loss.npy'.format(res_dir), hist.history["loss"])
    np.save('{}/val_loss.npy'.format(res_dir), hist.history["val_loss"])
    np.save("{}/test_loss.npy".format(res_dir), test_loss)
    #np.save("{}/test_mae.npy".format(res_dir), test_mae)
    np.save("{}/predicted_generation.npy".format(res_dir), pred_gen)
    np.save("{}/load.npy".format(res_dir), i_test)
    np.save("{}/true_generation.npy".format(res_dir), o_test)
    np.save("{}/pred_cost.npy".format(res_dir), pred_cost)
    np.save("{}/true_cost.npy".format(res_dir), test_cost)
    


    end_time = time.time()-start_time
    print('Job finished: Total Time Required was {}:{}'.format(int(end_time/60), int(end_time%60)))

if __name__ == '__main__':
    main()