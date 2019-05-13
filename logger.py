from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, iteration):
        """Log a scalar variable."""
        self.writer.add_scalar(
            tag=tag, scalar_value=value, global_step=iteration)

    def add_graph(self, model, image_size):
        """Log the computation graph of the given model using dummy input data."""
        dummy_input = torch.rand(2, 1, image_size, image_size)
        self.writer.add_graph(model, dummy_input, True)

    def add_image(self, tag, image, iteration):
        """Log an image."""
        self.writer.add_image(tag, image, iteration)

    def add_histogram(self, tag, array, iteration):
        """Log a histogram"""
        self.writer.add_histogram(tag, array, iteration)
