import torch
import torch.optim as optim
import torchvision
import lab_distribution as lab_dist
import plot
from custom_transforms import RGB2LAB, ToTensor
from network import Net
from logger import Logger


def get_ab_bins_dict(bin_path):
    """Given the path to the .bin file for training data,
    returns a dictionary with discretized color space bins.
    Keys in the dictionary: ab_bins, a_bins, b_bins."""
    return lab_dist.get_ab_bins_from_data(bin_path)


def get_transforms(ab_bins):
    """Given a list of discretized ab_bins, returns a list
    of transforms to be applied to the input data."""
    return torchvision.transforms.Compose([RGB2LAB(ab_bins), ToTensor()])


def get_dataloader(root, transform, batch_size, shuffle=True, num_workers=2):
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
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
    for i, data in enumerate(evalloader):
        inputs, _ = data
        lightness, z_truth, original = inputs['lightness'], inputs[
            'z_truth'], inputs['original_lab_image']

        optimizer.zero_grad()
        _ = model(lightness)
        loss = model.loss(z_truth)

        ab_outputs = model.decode_ab_values()
        colorized_im = torch.cat((lightness, ab_outputs), 1)

        # Log loss to tensorboardx
        logger.scalar_summary(prefix + '_loss_epoch_' + str(epoch), loss, i)

        # Log images to tensorboardx
        for j in range(colorized_im.detach().shape[0]):

            images = plot.imshow_torch(
                colorized_im.detach()[j, :, :, :], figure=0)
            if prefix == 'eval':
                logger.add_image(prefix + '_' + str(i) + '_' + str(j),
                                 torchvision.utils.make_grid(images), epoch)
            else:
                logger.add_image(prefix + '_epoch_' + str(epoch),
                                 torchvision.utils.make_grid(images), index)
            index += 1


def save_model_checkpoints(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path + '_' + str(epoch))


def create_model(bin_path,
                 ab_bins_dict,
                 pretrained_model_path,
                 log_dir,
                 learning_rate=.001,
                 betas=(.9, .999),
                 epsilon=1e-8,
                 weight_decay=.001,
                 mode='train'):
    # Create network
    cnn = Net(ab_bins_dict)
    cnn.get_rarity_weights(bin_path)

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
    logger.add_graph(cnn, image_size=96)
    return {
        'model': cnn,
        'optimizer': optimizer,
        'pretrained_epoch': pretrained_epoch,
        'logger': logger
    }


def test(pretrained_model_path, bin_path, test_dir, log_dir, batch_size,
         num_workers):
    # Create model
    ab_bins_dict = get_ab_bins_dict(bin_path)
    model = create_model(
        bin_path=bin_path,
        ab_bins_dict=ab_bins_dict,
        pretrained_model_path=pretrained_model_path,
        log_dir=log_dir)
    cnn, optimizer, pretrained_epoch, logger = model['model'], model[
        'optimizer'], model['pretrained_epoch'], model['logger']

    # Get training and test loaders
    transform = get_transforms(ab_bins_dict['ab_bins'])
    testloader = get_dataloader(
        root=test_dir,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    log_eval(
        cnn, optimizer, testloader, logger, pretrained_epoch, prefix='test')


def train(pretrained_model_path, bin_path, train_dir, eval_dir, eval_every_n,
          log_dir, log_every_n, checkpoint_dir, checkpoint_every_n, num_epochs,
          batch_size, num_workers, learning_rate, betas, epsilon,
          weight_decay):
    # Create model
    ab_bins_dict = get_ab_bins_dict(bin_path)
    model = create_model(
        bin_path=bin_path,
        ab_bins_dict=ab_bins_dict,
        pretrained_model_path=pretrained_model_path,
        log_dir=log_dir,
        learning_rate=learning_rate,
        betas=betas,
        epsilon=epsilon,
        weight_decay=weight_decay)
    cnn, optimizer, pretrained_epoch, logger = model['model'], model[
        'optimizer'], model['pretrained_epoch'], model['logger']

    # Get training and test loaders
    transform = get_transforms(ab_bins_dict['ab_bins'])
    trainloader = get_dataloader(
        root=train_dir,
        transform=transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    evalloader = get_dataloader(
        root=eval_dir,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.to(device)
    # Train!
    for epoch in range(pretrained_epoch, pretrained_epoch + num_epochs, 1):
        print('epoch = ', epoch)
        for i, data in enumerate(trainloader):
            print('i = ', i)
            inputs, labels = data
            lightness, z_truth, original = inputs['lightness'], inputs[
                'z_truth'], inputs['original_lab_image']

            optimizer.zero_grad()
            lightness = lightness.to(device)
            outputs = cnn(lightness)

            loss = cnn.loss(z_truth)
            loss.backward()
            optimizer.step()

        # Log info to tensorboardx
        if epoch % log_every_n == 0:
            ab_outputs = cnn.decode_ab_values()
            colorized_im = torch.cat((lightness, ab_outputs), 1)
            log_training_loss_and_image(logger, loss, colorized_im, epoch)

        # Log evaluation results to tensorboardx
        if epoch % eval_every_n == 0:
            log_eval(cnn, optimizer, evalloader, logger, epoch)

        # Store model checkpoint
        if epoch % checkpoint_every_n == 0:
            save_model_checkpoints(epoch, cnn, optimizer, loss, checkpoint_dir)
