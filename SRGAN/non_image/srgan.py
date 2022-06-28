"""
Code for the coefficient application.
"""
import numpy as np
import torch
from scipy.stats import norm, wasserstein_distance
from torch.utils.data import DataLoader
from recordclass import RecordClass
import matplotlib.pyplot as plt
from srgan import Experiment
from coefficient.data import ToyDataset
from coefficient.models import Generator, MLP, observation_count
from coefficient.presentation import generate_display_frame
from utility import seed_all, gpu, MixtureModel, standard_image_format_to_tensorboard_image_format


class CoefficientExperiment(Experiment):
    """The coefficient application."""

    def dataset_setup(self):
        settings = self.settings
        seed_all(settings.labeled_dataset_seed)
        self.train_dataset = ToyDataset(start=0, end=settings.labeled_dataset_size,
                                                  seed=self.settings.labeled_dataset_seed,
                                                  batch_size=settings.batch_size)
        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size, shuffle=True,
                                               pin_memory=self.settings.pin_memory,
                                               num_workers=settings.number_of_data_workers, drop_last=True)
        self.validation_dataset = ToyDataset(start=-settings.validation_dataset_size, end=None,
                                                       seed=self.settings.labeled_dataset_seed,
                                                       batch_size=settings.batch_size)
        unlabeled_dataset_start = settings.labeled_dataset_size + settings.validation_dataset_size
        if settings.unlabeled_dataset_size is not None:
            unlabeled_dataset_end = unlabeled_dataset_start + settings.unlabeled_dataset_size
        else:
            unlabeled_dataset_end = -settings.validation_dataset_size
        self.unlabeled_dataset = ToyDataset(start=unlabeled_dataset_start, end=unlabeled_dataset_end,
                                                      seed=self.settings.labeled_dataset_seed,
                                                      batch_size=settings.batch_size)
        self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size, shuffle=True,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers, drop_last=True)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.DNN = MLP(self.settings.hidden_size)
        self.D = MLP(self.settings.hidden_size)
        self.G = Generator(self.settings.hidden_size)

    def validation_summaries(self, step):
        """Prepares the summaries that should be run for the given application."""
        settings = self.settings
        dnn_summary_writer = self.dnn_summary_writer
        gan_summary_writer = self.gan_summary_writer
        DNN = self.DNN
        D = self.D
        G = self.G
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset
        # DNN training evaluation.
        self.evaluation_epoch(settings, DNN, train_dataset, dnn_summary_writer, '2 Train Error')
        # DNN validation evaluation.
        dnn_validation_mae = self.evaluation_epoch(settings, DNN, validation_dataset, dnn_summary_writer,
                                                   '1 Validation Error')
        # GAN training evaluation.
        self.evaluation_epoch(settings, D, train_dataset, gan_summary_writer, '2 Train Error')
        # GAN validation evaluation.
        self.evaluation_epoch(settings, D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_value=dnn_validation_mae)
        # Real images.
        train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
        train_iterator = iter(train_dataset_loader)
        examples, _ = next(train_iterator)
        #images_image = torchvision.utils.make_grid(to_image_range(examples[:9]), normalize=True, range=(0, 255), nrow=3)
        #gan_summary_writer.add_image('Real', examples.numpy())
        # Generated images.
        z = torch.randn(settings.batch_size, G.input_size).to(gpu)
        fake_examples = G(z).to('cpu')
        #fake_images_image = torchvision.utils.make_grid(to_image_range(fake_examples.data[:9]), normalize=True,
        #                                                range=(0, 255), nrow=3)
        #gan_summary_writer.add_image('Fake/Standard', fake_examples.numpy())
        z = torch.as_tensor(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]
                                         ).rvs(size=[settings.batch_size, G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = G(z).to('cpu')
        #fake_images_image = torchvision.utils.make_grid(to_image_range(fake_examples.data[:9]), normalize=True,
        #                                                range=(0, 255), nrow=3)
        #gan_summary_writer.add_image('Fake/Offset', fake_examples.numpy())

    class ComparisonValues(RecordClass):
        """A record class to hold the names of values which might be compared among methods."""
        mae: float
        mse: float
        rmse: float
        predicted_labels: np.ndarray

    def evaluation_epoch(self, settings, network, dataset, summary_writer, summary_name, comparison_value=None):
        """Runs the evaluation and summaries for the data in the dataset."""
        dataset_loader = DataLoader(dataset, batch_size=settings.batch_size)
        print(dataset)
        predicted_y, y = np.array([]), np.array([])
        for x, labels in dataset_loader:
            x =torch.tensor(x).to(gpu)
            batch_predicted_y = network(x)
            batch_predicted_y = batch_predicted_y.detach().to('cpu').view(-1).numpy()
            y = np.concatenate([y, labels])
            predicted_y = np.concatenate([predicted_y, batch_predicted_y])
        mae = np.abs(predicted_y - y).mean()
        summary_writer.add_scalar('{}/MAE'.format(summary_name), mae)
        nmae = mae / (y.max() - y.min())
        summary_writer.add_scalar('{}/NMAE'.format(summary_name), nmae)
        mse = (np.abs(predicted_y - y) ** 2).mean()
        summary_writer.add_scalar('{}/MSE'.format(summary_name), mse)
        fig, ax = plt.subplots()
        ax.plot(range(len(predicted_y[:100])),predicted_y[:100].reshape(-1), label="Predicted")
        ax.plot(range(len(y[:100])),y[:100].reshape(-1), label="True")
        ax.legend(loc='upper right')
        plt.title("Pumadyn Dataset FuzzyGAN Classification Injection: True RUL vs. Predicted RUL")
        plt.ylabel("Angle")
        plt.xlabel("Index")
        plt.savefig("FuzzyGANPumadyn_Classification_new.jpg")
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), mae / comparison_value)
        return mae
