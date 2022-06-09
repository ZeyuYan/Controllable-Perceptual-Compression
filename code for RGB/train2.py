"""
Learnable generative compression model modified from [1], 
implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
    arXiv:2006.09965 (2020).
"""
import numpy as np
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict

import torch, pdb
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

# Custom modules
from src.model import Model
from src.helpers import utils, datasets
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

#from srresnet_depU import _NetD
from srresnet_unet_CAB import _NetG, _NetD

# go fast boi!!
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger, storage, storage_test):

    start_time = time.time()
    model = Model(args, logger, storage, storage_test, model_type=args.model_type)
    logger.info(model)
    logger.info('Trainable parameters:')

    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model

def optimize_loss(loss, opt, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    opt.step()
    opt.zero_grad()

def test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage, best_test_loss, 
         start_time, epoch_start_time, logger, train_writer, test_writer):

    model.eval()  
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))
    
        compression_loss = losses['compression'] 
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)
        
        best_test_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(), 
                                     best_test_loss, start_time, epoch_start_time, 
                                     batch_size=data.shape[0], avg_bpp=test_bpp.mean().item(),header='[TEST]', 
                                     logger=logger, writer=test_writer)
        
    return best_test_loss, epoch_test_loss


def train(args, model, gener, discr, train_loader, test_loader, device, logger, G_optimizer, D_optimizer):

    start_time = time.time()
    test_loader_iter = iter(test_loader)
    current_D_steps, train_generator = 0, True
    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf     
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    test_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'test'))
    storage, storage_test = model.storage_train, model.storage_test


    for epoch in trange(args.n_epochs, desc='Epoch'):

        epoch_loss, epoch_test_loss = [], []  
        epoch_start_time = time.time()
        mse = []
        mse_ref=[]
        Dloss = []
        
        model.eval()
        iteration = 0

        for idx, (data, bpp) in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)
            
            try:
                if model.use_discriminator is True:
                    # Train D for D_steps, then G, using distinct batches
                    with torch.no_grad():
                        reconstruction,_ = model.compression_forward(data, fixE=True)
                    randx = int(np.random.rand()*0)
                    randy = int(np.random.rand()*0)
                    target = data[:,:,randx:randx+128,randy:randy+128]
                    mse_rec = reconstruction.reconstruction[:,:,randx:randx+128,randy:randy+128].detach()
                    latent = reconstruction.latents_quantized
                    #input = target + torch.randn(target.size()).cuda()*0.1
                    iteration = iteration + 1

                    discr.zero_grad()

                    D_result = discr(target, mse_rec).squeeze()
                    D_real_loss = -D_result.mean()

                    with torch.no_grad():
                        G_result = gener(latent)
                    D_result = discr(G_result.data, mse_rec).squeeze()

                    D_fake_loss = D_result.mean()

                    D_train_loss = D_real_loss + D_fake_loss
                    Dloss.append(D_train_loss.data)

                    D_train_loss.backward()
                    D_optimizer.step()

                    #gradient penalty
                    discr.zero_grad()
                    alpha = torch.rand(target.size(0), 1, 1, 1)
                    alpha1 = alpha.cuda().expand_as(target)
                    interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * G_result.data, requires_grad=True)
                    interpolated2 = Variable(mse_rec, requires_grad=True)
                    
                    out = discr(interpolated1, interpolated2).squeeze()

                    grad = torch.autograd.grad(outputs=out,
                                            inputs=interpolated1,
                                            grad_outputs=torch.ones(out.size()).cuda(),
                                            retain_graph=True,
                                            create_graph=True,
                                            only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    #grad_l2norm[grad_l2norm<0] = 0
                    d_loss_gp = torch.mean((grad_l2norm - 0) ** 2)

                    # Backward + Optimize
                    gp_loss = 10 * d_loss_gp

                    gp_loss.backward()
                    D_optimizer.step()

                    # train generator G
                    discr.zero_grad()
                    gener.zero_grad()

                    G_result = gener(latent)
                    D_result = discr(G_result, mse_rec).squeeze()

                    #mse_low = torch.mean((dsmpl(G_result) - dsmpl(target))**2)

                    mse_loss = torch.mean((G_result - target)**2)
                    mse_reference = torch.mean((mse_rec - target)**2)
                    move_loss = torch.sqrt(torch.sum((G_result - mse_rec)**2, dim=[1,2,3]))
                    #move_loss = torch.mean((G_result - input)**2)
                    mse.append(mse_loss.data)
                    mse_ref.append(mse_reference.data)

                    G_train_loss = - D_result.mean() + 0.005 * move_loss.mean()
                    #G_train_loss = move_loss.mean()

                    G_train_loss.backward()
                    G_optimizer.step()

            except KeyboardInterrupt:
                # Note: saving not guaranteed!
                if model.step_counter > args.log_interval+1:
                    logger.warning('Exiting, saving ...')
                    return model, ckpt_path
                else:
                    return model, None

            if iteration % args.log_interval == 1:
                #print("===> Epoch[{}]({}): Loss_img: {:.5}, Loss_D: {:.5}".format(epoch, iteration, mse_loss.data, D_train_loss.data))
                print("===> Epoch[{}]({}): Loss_img: {:.5}, Ref_img: {:.5}, Loss_D: {:.5}".format(epoch, iteration, torch.mean(torch.FloatTensor(mse)),
                                    torch.mean(torch.FloatTensor(mse_ref)), torch.mean(torch.FloatTensor(Dloss))))
                saveflag=1
                for subidx in range(G_result.shape[0]):
                    fname = os.path.join('./data/reconstructions', "{}.png".format(saveflag))
                    torchvision.utils.save_image(G_result[subidx], fname, normalize=False)
                    fname = os.path.join('./data/reconstructions', "{}_inp.png".format(saveflag))
                    torchvision.utils.save_image(mse_rec[subidx], fname, normalize=False)
                    saveflag+=1
    
        save_checkpoint(gener, discr, epoch)

        # End epoch
        #mean_epoch_loss = np.mean(epoch_loss)
        #mean_epoch_test_loss = np.mean(epoch_test_loss)

        #logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch, 
        #    mean_epoch_loss, mean_epoch_test_loss))

        if model.step_counter > args.n_steps:
            break
    
    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
    #args.ckpt = ckpt_path
    save_checkpoint(gener, discr, epoch)
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), model.step_counter))
    
    return model, ckpt_path

def save_checkpoint(gener, discr, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": gener, "discr": discr}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':

    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN), 
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='low', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=hific_args.log_interval, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=hific_args.save_interval, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=hific_args.batch_size, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")
    general.add_argument("--resume", default="./checkpoint/model_epoch_39.pth", type=str, help="Path to checkpoint (default: none)")

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=hific_args.n_steps, 
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=hific_args.n_epochs, 
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=hific_args.learning_rate, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=hific_args.weight_decay, help="Coefficient of L2 regularization.")

    # Architecture-related options
    arch_args = parser.add_argument_group("Architecture-related options")
    arch_args.add_argument('-lc', '--latent_channels', type=int, default=hific_args.latent_channels,
        help="Latent channels of bottleneck nominally compressible representation.")
    arch_args.add_argument('-nrb', '--n_residual_blocks', type=int, default=hific_args.n_residual_blocks,
        help="Number of residual blocks to use in Generator.")

    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--warmstart_ckpt", default=None, help="Path to autoencoder + hyperprior ckpt.")

    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args

    start_time = time.time()
    device = utils.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)

    storage = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))

    if args.warmstart is True:
        assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
        logger.info('Warmstarting discriminator/generator from autoencoder/hyperprior model.')
        if args.model_type != ModelTypes.COMPRESSION_GAN:
            logger.warning('Should warmstart compression-gan model.')
        args, model, _= utils.load_model(args.warmstart_ckpt, logger, device, 
            model_type=args.model_type, current_args_d=dictify(args), strict=False, prediction=False)
        
        gener = _NetG()
        discr = _NetD()
        gener = gener.cuda()
        discr = discr.cuda()
        if cmd_args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(cmd_args.resume))
                checkpoint = torch.load(cmd_args.resume)
                gener.load_state_dict(checkpoint["model"].state_dict())
                discr.load_state_dict(checkpoint["discr"].state_dict())
        G_optimizer = torch.optim.RMSprop(gener.parameters(), lr=args.learning_rate/2)
        D_optimizer = torch.optim.RMSprop(discr.parameters(), lr=args.learning_rate)
        
    else:
        model = create_model(args, device, logger, storage, storage_test)
        model = model.to(device)
        amortization_parameters = itertools.chain.from_iterable(
            [am.parameters() for am in model.amortization_models])

        hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

        #amortization_opt = torch.optim.Adam(amortization_parameters, lr=args.learning_rate)
        amortization_opt = torch.optim.RMSprop(amortization_parameters, lr=args.learning_rate)
        hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters, lr=args.learning_rate)
        optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

        if model.use_discriminator is True:
            discriminator_parameters = model.Discriminator.parameters()
            #disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate * 10)
            disc_opt = torch.optim.RMSprop(discriminator_parameters, lr=args.learning_rate*10)
            optimizers['disc'] = disc_opt

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        # Not supported at this time
        raise NotImplementedError('MultiGPU not supported yet.')
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))
    logger.info('USING DATASET: {}'.format(args.dataset))

    test_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='validation',
                                shuffle=True,
                                normalize=args.normalize_input_image)

    train_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='train',
                                shuffle=True,
                                normalize=args.normalize_input_image)

    args.n_data = len(train_loader.dataset)
    args.image_dims = train_loader.dataset.image_dims
    logger.info('Training elements: {}'.format(args.n_data))
    logger.info('Input Dimensions: {}'.format(args.image_dims))
    #logger.info('Optimizers: {}'.format(optimizers))
    logger.info('Using device {}'.format(device))

    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    logger.info(metadata)

    """
    Train
    """
    model, ckpt_path = train(args, model, gener, discr, train_loader, test_loader, device, logger, G_optimizer, D_optimizer)

    """
    TODO
    Generate metrics
    """
