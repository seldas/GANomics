"""General-purpose training script for image-to-image translation."""
import os, re, time, argparse
import pandas as pd
from options.train_options import TrainOptions
from data import create_dataset
from model import create_model
from util.visualizer import Visualizer

EXPLOSION_FACTOR = 5.0        # stop & reload if current loss > this * best loss
LATEST_SAVE_FREQ = 10         # save "latest" every 10 epochs as requested
save_i = 1 # initial saving

parser = argparse.ArgumentParser(description="Description of your program.")

def total_loss_from_dict(loss_dict):
    # sum all reported losses; robust to mixed tensor/float
    return float(sum(float(v) for v in loss_dict.values()))

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options

    # manual overrides from your original script
    t = opt.size
    r = opt.run
    ckp_dir = opt.ckp_dir
    opt.datatype = 'text'
    opt.checkpoints_dir = ckp_dir
    opt.name = 'SampleSize_' + str(t) + '_' + str(r)
    opt.netD = 'fullyconnected'
    opt.netG = 'fullyconnected'
    opt.display_id = 0  # disable visdom

    n = opt.network
    if n == 'noFEEDBACK':
        opt.model = 'trans_cycle_gan'
        opt.lambda_median = 0
        opt.lambda_A = 10.0
        opt.lambda_B = 10.0
        opt.lambda_C = 10.0
        opt.lambda_D = 10.0
    if n == 'FEEDBACKv2':
        opt.model = 'trans_cycle_gan'
        opt.lambda_median = 1.0
        opt.lambda_A = 10.0
        opt.lambda_B = 10.0
        opt.lambda_C = 10.0
        opt.lambda_D = 10.0
    else:
        opt.model = 'cycle_gan'

    opt.serial_batches = (n != 'REGULAR')

    # build
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    # print('The number of training images = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)  # creates schedulers, may load if --continue_train

    visualizer = Visualizer(opt)
    total_iters = 0

    # === Robust training state ===
    max_epochs_planned = opt.n_epochs + opt.n_epochs_decay
    current_epoch = opt.epoch_count
    total_epoch_attempts = 0         # counts *all* attempts, including aborted ones
    best_total_loss = {}   # best observed running loss (sum of reported losses)
    last_stable_epoch = ((current_epoch - 1) // LATEST_SAVE_FREQ) * LATEST_SAVE_FREQ  # last multiple of 10

    # main training while-loop so we can retry the same epoch after reloads
    while current_epoch <= max_epochs_planned and total_epoch_attempts < (opt.n_epochs + opt.n_epochs_decay):
        total_epoch_attempts += 1
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        exploded = False
        visualizer.reset()

        # print(f'=== Epoch {current_epoch}/{max_epochs_planned} (attempt {total_epoch_attempts}) ===')

        # inner loop
        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(current_epoch, epoch_iter, losses, t_comp, t_data)

                # monitor for explosion
                if n == 'FEEDBACKv2':
                    loss_flag = 0
                    for k, v in losses.items():
                        if k in ['median_A', 'median_B','cycle_A','cycle_B'] and best_total_loss.get(k, float('inf')) <= EXPLOSION_FACTOR * v:
                            loss_flag = 1
                            break
                    # curr_total_loss = losses['median_B'] + losses['median_A']
                    if loss_flag == 0 :
                        best_total_loss = losses
                    else:
                        exploded = True
                        break

            iter_data_time = time.time()

        # if exploded, reload and retry the *same* epoch number
        if exploded:
            # reload last "latest" checkpoint (networks only)
            try:
                model.load_networks('latest')  # provided by BaseModel
                continue  # do NOT bump epoch; we’ll retry this epoch number
            except:
                continue # in rare case the last saved model is not available, skip the exploded checking;

        # epoch finished successfully – save, then update LR, then bump epoch
        if n == 'FEEDBACKv2':
            if current_epoch % LATEST_SAVE_FREQ == 0:
                # save numbered snapshot + rolling "latest"
                model.save_networks('latest')
                last_stable_epoch = current_epoch
                # print(f'[CKPT] Saved checkpoints for epoch {current_epoch} (and updated "latest").')
            
            if total_epoch_attempts // opt.save_epoch_freq >= save_i: 
                model.save_networks(total_epoch_attempts)
                save_i += 1
        else: # not GANomics, use default setting
            if total_epoch_attempts // opt.save_epoch_freq >= save_i:   # cache our model every <save_epoch_freq> epochs
                model.save_networks('latest')
                model.save_networks(total_epoch_attempts)
                save_i += 1
            
        # step schedulers only after a *successful* epoch
        model.update_learning_rate()

        # advance epoch
        current_epoch += 1

    # training done (either completed planned epochs or hit total-attempts budget)
    # ensure we leave the last stable checkpoint as "latest"
    # print('[END] Training finished. Restoring/saving last stable checkpoint...')
    model.load_networks('latest')   # ensures nets are at last stable state
    model.save_networks('latest')   # overwrite to be safe
