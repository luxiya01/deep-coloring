import os
import math
import numpy as np
import torch
import torch.optim as optim
import torchvision
import lab_distribution as lab_dist
import plot
from custom_transforms import RGB2LAB, ToTensor
from network import Net
from logger import Logger
from functools import partial


def get_prior_bins_dict(bin_path, outfile):
    """Given the path to the .bin file for training data,
    returns a dictionary with discretized color space bins
    and probability distributions. Also store the output in
    a .npz file.
    Keys in the dictionary: ab_bins, a_bins, b_bins, w_bins."""
    return lab_dist.get_and_store_ab_bins_and_rarity_weights(
        data_dir=bin_path, outfile=outfile)


def read_prior_bins_dict(npz_path):
    return np.load(npz_path)


def get_transforms(ab_bins):
    """Given a list of discretized ab_bins, returns a list
    of transforms to be applied to the input data."""
    return torchvision.transforms.Compose([RGB2LAB(ab_bins, 'hsl'), ToTensor()])


def get_image_dataloader(root,
                         transform,
                         batch_size,
                         shuffle=True,
                         num_workers=2):
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


def get_tensor_dataloader(root, batch_size, shuffle=True, num_workers=2):
    loader = partial(torch.load)
    extensions = ['pt']
    dataset = torchvision.datasets.DatasetFolder(
        root=root, loader=loader, extensions=extensions)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


def get_logger(log_dir):
    return Logger(log_dir)


def log_training_loss_and_image(logger, loss, colorized_im, epoch):
    # Log loss to tensorboardx
    logger.scalar_summary('loss', loss, epoch)

    # Log images to tensorboardx
    for j in range(colorized_im.detach().shape[0]):
        images = plot.imshow_torch(colorized_im.detach()[j, :, :, :], figure=0)
        logger.add_image('train_' + str(j),
                         torchvision.utils.make_grid(images), epoch)


def log_eval(model, optimizer, evalloader, logger, epoch, prefix='eval'):
    index = 0
    losses = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i, data in enumerate(evalloader):
            inputs, _ = data
            lightness, z_truth = inputs['lightness'].to(device), inputs['z_truth'].to(device)

            optimizer.zero_grad()
            _ = model(lightness)
            loss = model.loss(z_truth)
            losses.append(loss)

            # Log images to tensorboardx if prefix is not eval, i.e. not training
            if prefix != 'eval':
                ab_outputs = model.decode_ab_values()
                colorized_im = torch.cat((lightness, ab_outputs), 1).cpu()

                for j in range(colorized_im.detach().shape[0]):

                    images = plot.imshow_torch(
                        colorized_im.detach()[j, :, :, :], figure=0)
                    logger.add_image(prefix + '_epoch_' + str(epoch),
                                     torchvision.utils.make_grid(images), index)
                    index += 1

        # Log average loss to tensorboardx
        losses = torch.FloatTensor(losses)
        logger.scalar_summary(prefix + '_average_loss', losses.mean(), epoch)
        logger.histogram_summary(prefix + '_loss_hist', losses, epoch)


def save_model_checkpoints(epoch, model, optimizer, loss, path):
    checkpoint_path = os.path.join(
        os.path.join(os.getcwd(), path), 'checkpoint_' + str(epoch) + '.pth')
    print(checkpoint_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)


def create_model(pretrained_model_path,
                 log_dir,
                 bin_path='',
                 npz_path='',
                 learning_rate=.001,
                 betas=(.9, .999),
                 epsilon=1e-8,
                 weight_decay=.001,
                 mode='train'):
    print('mode = ', mode)
    if bin_path != '':
        bins_dict = get_prior_bins_dict(bin_path,
                                        'prior_distribution_' + mode + '.npz')
    else:
        bins_dict = read_prior_bins_dict(npz_path)

    # Create network
    cnn = Net(bins_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.to(device)

    cnn.set_rarity_weights(bins_dict['w_bins'])

    # Define criterion and optimizer for gradient descent
    optimizer = optim.Adam(
        cnn.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=epsilon,
        weight_decay=weight_decay)

    pretrained_epoch = 1
    # Load pretrained model!
    if pretrained_model_path != '':
        checkpoint = torch.load(pretrained_model_path)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        pretrained_epoch += checkpoint['epoch']

        if mode == 'train':
            cnn.train()
        else:
            cnn.eval()

    # Log computation graph to tensorboardx
    logger = get_logger(log_dir)
    # logger.add_graph(cnn, image_size=96)
    return {
        'model': cnn,
        'optimizer': optimizer,
        'pretrained_epoch': pretrained_epoch,
        'logger': logger,
        'bins_dict': bins_dict
    }


def test(pretrained_model_path,
         test_dir,
         log_dir,
         batch_size,
         num_workers,
         bin_path='',
         npz_path=''):
    # Create model
    model = create_model(
        bin_path=bin_path,
        npz_path=npz_path,
        pretrained_model_path=pretrained_model_path,
        log_dir=log_dir,
        mode='test')
    cnn, optimizer, pretrained_epoch, logger, bins_dict = model[
        'model'], model['optimizer'], model['pretrained_epoch'], model[
            'logger'], model['bins_dict']

    # Get training and test loaders
    transform = get_transforms(bins_dict['ab_bins'])
    testloader = get_tensor_dataloader(
        root=test_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    print('Log eval......')

    log_eval(
        cnn, optimizer, testloader, logger, pretrained_epoch, prefix='test')


def train(pretrained_model_path,
          train_dir,
          eval_dir,
          eval_every_n,
          log_dir,
          log_every_n,
          checkpoint_dir,
          checkpoint_every_n,
          num_epochs,
          batch_size,
          num_workers,
          learning_rate,
          betas,
          epsilon,
          weight_decay,
          bin_path='',
          npz_path=''):
    # Create model
    model = create_model(
        bin_path=bin_path,
        npz_path=npz_path,
        pretrained_model_path=pretrained_model_path,
        log_dir=log_dir,
        learning_rate=learning_rate,
        betas=betas,
        epsilon=epsilon,
        weight_decay=weight_decay)
    cnn, optimizer, pretrained_epoch, logger, bins_dict = model[
        'model'], model['optimizer'], model['pretrained_epoch'], model[
            'logger'], model['bins_dict']

    # Get training and test loaders
    #    transform = get_transforms(bins_dict['ab_bins'])
    #    trainloader = get_image_dataloader(
    #        root=train_dir,
    #        transform=transform,
    #        batch_size=batch_size,
    #        shuffle=True,
    #        num_workers=num_workers)
    #    evalloader = get_image_dataloader(
    #        root=eval_dir,
    #        transform=transform,
    #        batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers)

    # Get training and test tensor loaders
    trainloader = get_tensor_dataloader(
        root=train_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    evalloader = get_tensor_dataloader(
        root=eval_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    # Train!
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    individual_losses = np.zeros(
        max(1, math.ceil(len(trainloader.dataset) / batch_size)))
    for epoch in range(pretrained_epoch, pretrained_epoch + num_epochs, 1):
        print('epoch = ', epoch)
        for i, data in enumerate(trainloader):
            inputs, labels = data
            lightness, z_truth = inputs['lightness'].to(
                device), inputs['z_truth'].to(device)

            optimizer.zero_grad()

            outputs = cnn(lightness)

            loss = cnn.loss(z_truth)
            individual_losses[i] = loss
            loss.backward()
            optimizer.step()

        # Log info to tensorboardx
        if epoch % log_every_n == 0:
            ab_outputs = cnn.decode_ab_values()
            colorized_im = torch.cat((lightness, ab_outputs), 1)
            log_training_loss_and_image(logger, np.mean(individual_losses),
                                        colorized_im.cpu(), epoch)
            logger.histogram_summary(
                'Individual training loss',
                torch.FloatTensor(individual_losses).cpu(),
                epoch,
                bins=100)

        # Log evaluation results to tensorboardx


#        if epoch % eval_every_n == 0:
#            log_eval(cnn, optimizer, evalloader, logger, epoch)

# Store model checkpoint every n epochs and at the last epoch
        if epoch % checkpoint_every_n == 0 or epoch == pretrained_epoch + num_epochs - 1:
            save_model_checkpoints(epoch, cnn, optimizer, loss, checkpoint_dir)
