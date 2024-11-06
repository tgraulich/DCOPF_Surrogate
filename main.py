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
    argparser.add_argument('-w', '--weight', default=0, type=float, help="Weight with which constraint violation get added to the loss function")
    argparser.add_argument('-eps', '--epsilon', default=1e-3, type=float, help="Bound below which negative values of violation contribute to loss function")
    argparser.set_defaults(debug_load=False, baseline=True, schedule=False)

    args = argparser.parse_args()
 
    return args

def mae_loss(divider):
    def _mae_loss(ytrue, ypred):
        ytrue = ytrue[:,:divider]
        ypred = ypred[:,:divider]
        return K.mean(tf.abs(ytrue - ypred))
    return _mae_loss

def mape_loss(divider):
    def _mape_loss(ytrue, ypred):
        ytrue = ytrue[:,:divider]
        ypred = ypred[:,:divider]
        return K.mean(tf.divide(tf.abs(ytrue - ypred),ytrue))
    return _mape_loss

def system_cost(cost, divider, pmax, pmin):
    def _system_cost(ytrue, ypred):
        pred_scale = ypred[:,:divider]
        load = ytrue[:,divider:]
        pred_gen = scale_to_gen(pred_scale, pmax[1:], pmin[1:])
        pred_gen = get_slack_bus_gen(pred_gen, load)
        pred_cost = calculate_cost(pred_gen, cost)
        return K.mean(pred_cost)
    return _system_cost

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

def slack_bus_violation(divider, pmax, pmin):
    def _slack_bus_violation(ytrue, ypred):
        pred_scale = ypred[:,:divider]
        load = ytrue[:,divider:]
        pred_gen = scale_to_gen(pred_scale, pmax[1:], pmin[1:])
        return K.max(tf.reduce_max(tf.pad(get_slack_bus_gen2(pred_gen, load)-pmax[0], [[0,0],[1,0]]), axis=1))
    return _slack_bus_violation

def load_balance(divider, pmax, pmin):
    def _load_balance(ytrue, ypred):
        true_scale = ytrue[:,:divider]
        pred_scale = ypred[:,:divider]
        load = ytrue[:,divider:]
        true_gen = scale_to_gen(true_scale, pmax[1:], pmin[1:])
        pred_gen = scale_to_gen(pred_scale, pmax[1:], pmin[1:])
        true_gen = get_slack_bus_gen(true_gen, load)
        pred_gen = get_slack_bus_gen(pred_gen, load)
        return K.max(tf.reduce_sum(true_gen, axis=1)-tf.reduce_sum(pred_gen, axis=1))
    return _load_balance

def combined_loss(cost, divider, pmax, pmin, weight):
    def _combined_loss(ytrue, ypred):
        return (1-weight)*mae_loss(divider)(ytrue, ypred)+weight*cost_metric(cost, divider, pmax, pmin)(ytrue, ypred)
    return _combined_loss

def main():

    start_time = time.time()
    config = parse_args()

    learning_rate    = config.learning_rate
    num_epochs       = config.epochs
    batch_size       = config.batch_size
    l2_scale         = config.l2_scale
    weight           = config.weight
    eps              = config.epsilon
    res_dir = config.res_dir
    if config.debug_load:
        print("Running in Debug mode (2 epochs)...")
        num_epochs = 2
        res_dir="results/res_"



    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    f = open('{}/summary.txt'.format(res_dir), 'w')
    f.write('##### Summary: #####\n\n')
    f.write('### Hyperparameters: ###\n')
    f.writelines('\n'.join(['Random Key: {}'.format(config.key), 'Learning Rate: {}'.format(learning_rate), 'Batch Size: {}'.format(batch_size), 'L2 Scale: {}'.format(l2_scale), 'Weight: {}'.format(weight)]))
    f.close()
    print('Hyperparameters:\n Learning Rate: {}\n Number of Epochs: {}\n Batch Size: {}\n L2 Scale: {}\n'.format(learning_rate, num_epochs, batch_size, l2_scale))

    print("Loading Data...")
    i_dir = config.data_dir

    data = loadmat(i_dir)

    load = np.array(data["load_samples_full"])
    base_load=np.squeeze(data["load"])
    non_active_load = np.argwhere(base_load == 0)
    base_load = np.delete(base_load, non_active_load)
    load = np.delete(load, non_active_load, axis=1)
    cost=np.array(data["cost"])
    gen=np.array(data["generator_samples"])
    non_active_gen = np.argwhere(np.all(gen[..., :] <= 1e-2, axis=0))
    gen = np.delete(gen, non_active_gen, axis=1)
    cost = np.delete(cost, non_active_gen, axis=0)
    divider = gen.shape[1]-1
    pmin=np.delete(np.array(data["pmin"]), non_active_gen, axis=0)
    pmax=np.delete(np.array(data["pmax"]), non_active_gen, axis=0)
    #true_costs = np.array(data["objective"])
    x= data["sampling_range"]
    print("Plotting Data...")

    '''Plot generator distributions'''
    rows, cols = find_factors(gen.shape[1])
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.hist(gen[:,i], bins=20, histtype="step", color="blue")
        ax.set_title("Generator {}".format(i+1))
        ax.set_xlabel("Dispatch / MW")

    fig.savefig("{}/generators.png".format(res_dir), bbox_inches="tight", dpi=300)

    '''Plot cost distribution'''
    fig = plt.figure()

    plt.hist(calculate_cost(gen, cost), bins=20, color="blue", histtype="step")
    plt.xlabel("System Cost / [$/h]")

    fig.savefig("{}/costs.png".format(res_dir), dpi=300)

    '''Plot load distributions'''
    rows, cols = find_factors(load.shape[1])
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.hist(load[:,i], bins=20, histtype="step", color="blue")
        ax.set_title("Load {}".format(i+1))
        ax.set_xlabel("Demand / MW")

    fig.savefig("{}/loads.png".format(res_dir), bbox_inches="tight", dpi=300)

    print("Preparing data for training...")

    i_train, i_test, o_train, o_test = train_test_split(load, gen, test_size=0.1, random_state=config.key)

    train_cost = calculate_cost(o_train, cost)
    test_cost = calculate_cost(o_test, cost)

    scales_train = gen_to_output(o_train, pmax, pmin)[:,1:]
    scales_test = gen_to_output(o_test, pmax, pmin)[:,1:]

    output_train=np.append(scales_train, i_train, axis=1)
    output_test=np.append(scales_test, i_test, axis=1)

    input_train = load_to_scale(i_train, base_load, x)
    input_test = load_to_scale(i_test, base_load, x)
    
    print("Input shape: {}, Output shape: {}".format(input_train.shape, output_train.shape))
    print("Plotting scaled data")

    '''Plot generator distributions'''
    rows, cols = find_factors(scales_train.shape[1])
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.hist(scales_train[:,i], bins=20, histtype="step", color="blue")
        ax.set_title("Generator {}".format(i+1))

    fig.savefig("{}/generators_scaled.png".format(res_dir), bbox_inches="tight", dpi=300)

    '''Plot load distributions'''
    rows, cols = find_factors(input_train.shape[1])
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.hist(input_train[:,i], bins=20, histtype="step", color="blue")
        ax.set_title("Load {}".format(i+1))

    fig.savefig("{}/loads_scaled.png".format(res_dir), bbox_inches="tight", dpi=300)

    print("Starting training...")

    tf.keras.backend.set_floatx('float64')
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_train.shape[1],)))
    model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_scale)))
    model.add(tf.keras.layers.Dense(units=output_train.shape[1], activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=mae_loss(divider), optimizer=opt, metrics=[mae_loss(divider), mape_loss(divider), cost_metric(cost, divider, pmax, pmin), system_cost(cost, divider, pmax, pmin), slack_bus_violation(divider, pmax, pmin), load_balance(divider, pmax, pmin)])
    hist = model.fit(input_train, output_train, verbose=1, epochs=num_epochs, validation_split=0.05, batch_size=batch_size)
    print("Evaluating model...")

    predictions = model.predict(input_test)[:,:divider]
    pred_gen = np.array(scale_to_gen(predictions, pmax[1:], pmin[1:]))
    pred_gen = get_slack_bus_gen(pred_gen, i_test)
    pred_cost = calculate_cost(pred_gen, cost)

    test_loss, test_mae, test_mape, test_cost_metric, test_system_cost, test_slack_violation, test_balance  = model.evaluate(input_test, output_test)

    print("Saving & plotting results")

    '''Plotting loss'''
    fig = plt.figure()
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.legend(loc=1)
    fig.savefig("{}/loss.png".format(res_dir), dpi=300)

    '''Plotting generator distributions'''
    rows, cols = find_factors(pred_gen.shape[1])
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.hist(pred_gen[:,i], bins=20, histtype="step", label="Prediction", linestyle="--", color="blue")
        ax.hist(o_test[:,i], bins=20, histtype="step", label="True", linestyle="--", color="red")
        ax.set_title("Generator {}".format(i+1))
        ax.set_xlabel("Dispatch / MW")
        if i == 0:
            ax.legend(loc=(-1.5,0.6))

    fig.savefig("{}/gen_predictions.png".format(res_dir), bbox_inches="tight", dpi=300)

    '''Plotting cost distribution'''

    fig = plt.figure()
    plt.hist(test_cost, bins=20, histtype="step", label="True", linestyle="--", color="red")
    plt.hist(pred_cost, bins=20, histtype="step", label="Prediction", linestyle="--", color="blue")
    plt.legend(loc=1)
    plt.xlabel("System Cost / [$/h]")
    fig.savefig("{}/pred_cost.png".format(res_dir), dpi=300)

    '''Plotting example generator dispatches'''

    fig, axs = plt.subplots(3,3, figsize = (20,15))

    axs = axs.flatten()
    idx = np.random.randint(0, len(pred_gen), len(axs))
    for i, ax in enumerate(axs):
        ax.bar(np.arange(o_test.shape[1])-0.1+1, o_test[idx[i]], width=0.2, label="Truth")
        ax.bar(np.arange(o_test.shape[1])+0.1+1, pred_gen[idx[i]],width=0.2, label="Prediction")
        ax.set_title("Sample {}".format(idx[i]))
        ax.set_ylabel("Dispatch / MW")
        if i == 2:
            ax.legend(loc=(1.1,0.6))
    fig.savefig("{}/dispatch_examples.png".format(res_dir), dpi=300)

    np.save('{}/train_loss.npy'.format(res_dir), hist.history["loss"])
    np.save('{}/train_mae.npy'.format(res_dir), hist.history["_mae_loss"])
    np.save('{}/train_mape.npy'.format(res_dir), hist.history["_mape_loss"])
    np.save('{}/train_cost_metric.npy'.format(res_dir), hist.history["_cost_metric"])
    np.save('{}/train_system_cost.npy'.format(res_dir), hist.history["_system_cost"])
    np.save('{}/train_slack_violation.npy'.format(res_dir), hist.history["_slack_bus_violation"])
    np.save('{}/train_balance.npy'.format(res_dir), hist.history["_load_balance"])
    np.save('{}/val_loss.npy'.format(res_dir), hist.history["val_loss"])
    np.save("{}/test_loss.npy".format(res_dir), test_loss)
    np.save("{}/test_mae.npy".format(res_dir), test_mae)
    np.save("{}/test_mape.npy".format(res_dir), test_mape)
    np.save("{}/test_cost_metric.npy".format(res_dir), test_cost_metric)
    np.save("{}/test_system_cost.npy".format(res_dir), test_system_cost)
    np.save("{}/test_slack_violation.npy".format(res_dir), test_slack_violation)
    np.save("{}/test_balance.npy".format(res_dir), test_balance)
    np.save("{}/predicted_generation.npy".format(res_dir), pred_gen)
    np.save("{}/load.npy".format(res_dir), i_test)
    np.save("{}/true_generation.npy".format(res_dir), o_test)
    np.save("{}/pred_cost.npy".format(res_dir), pred_cost)
    np.save("{}/true_cost.npy".format(res_dir), test_cost)
    


    end_time = time.time()-start_time
    print('Job finished: Total Time Required was {}:{}'.format(int(end_time/60), int(end_time%60)))

if __name__ == '__main__':
    main()