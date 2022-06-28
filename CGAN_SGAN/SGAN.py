import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# %matplotlib inline
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import Image

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import time

import dataset, metrics, config

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new2 results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def conjunction(a, b):
    conj = torch.mul(a.repeat(1, 2), b)
    return conj


def disjunction(a, b):
    disj = a + b - torch.mul(a, b)
    return disj


def implication(conj, disj, s=9, b0=-0.5):
    imp = 1 - conj + conj * disj.repeat(1, 2)
    sig_imp = (((1 + torch.exp(torch.as_tensor(s / 2))) * torch.sigmoid(s * (imp) - s / 2)) - 1) / (
            torch.exp(torch.as_tensor(s / 2)) - 1)
    return sig_imp


def aggregate(inputs):
    agg = torch.prod(inputs, dim=1)
    agg = abs(agg)
    return agg


def conjunction_gen(a, b):
    conj = torch.mul(a.repeat(1, 2), b)
    return conj


def disjunction_gen(a, b):
    disj = a + b - torch.mul(a, b)
    return disj


def implication_gen(conj, disj, s=9, b0=-0.5):
    imp = 1 - conj + conj * disj.repeat(1, 2)
    sig_imp = (((1 + torch.exp(torch.as_tensor(s / 2))) * torch.sigmoid(s * (imp) - s / 2)) - 1) / (
            torch.exp(torch.as_tensor(s / 2)) - 1)
    return sig_imp

def aggregate_gen(inputs):
    agg = torch.prod(inputs, dim=1)
    agg = abs(agg)
    return agg



scenario = "driving_torch"

dataset_config = config.DatasetConfig(scenario=scenario)

X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(scenario=dataset_config.scenario)
#from sklearn import preprocessing

#min_max_scaler = preprocessing.MinMaxScaler()

#y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
#y_test = min_max_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

#y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1)).reshape(-1)
#y_train = min_max_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)

print(max(y_train))
print(min(y_train))

print(max(y_test))
print(min(y_test))

#from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_valid = scaler.transform(X_valid)
#X_test = scaler.transform(X_test)
#tests = ["classification_5", "classification_10", "classification_15", "classification_20", "classification_25", "classification_30", "classification_35", "classification_40", "regression_5", "regression_10", "regression_15", "regression_20", "regression_25", "regression_30", "regression_35", "regression_40", "SGAN"]
tests = ["double_25_40"]
num_tests = len(tests)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels,
                                             kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1,
           batch_norm=True, bias=False):
    """Creates a transposed-convolutional layer, with optional batch
       normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                     stride, padding, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.Linear1 = nn.Linear(nz, 128*8*8)
        self.convT1 = deconv(128, 128, 4, 2, 2)
        self.convT2 = deconv(128, 128, 4, 3, 0)
        self.output = deconv(128, nc, 4, 3, 1, batch_norm=False)

    def forward(self, input):
        out = F.leaky_relu(self.Linear1(input.squeeze()), inplace=True)
        out = out.reshape(-1, 128, 8, 8)
        out = F.leaky_relu(self.convT1(out), inplace=True)
        out = F.leaky_relu(self.convT2(out), inplace=True)
        out = self.output(out)
        #print(out.shape)
        return out
class Discriminator(nn.Module):
    def __init__(self, ngpu, out_1, out_2):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        #self.num_classes = num_classes
        self.conv1 = conv(nc, 128, 9, 6, 1)
        self.conv2 = conv(128, 64, 9, 6, 1)
        self.Linear = nn.Linear(576, out_1)
        self.Linear2 = nn.Linear(576, out_2)

    def forward(self, input):
        out = F.leaky_relu(self.conv1(input), inplace=True)
        out = F.leaky_relu(self.conv2(out), inplace=True)
        out = out.view(-1, 576)
        out1 = self.Linear(out)
        out2 = self.Linear2(out)
        adv = torch.sigmoid(out1.squeeze())
        aux = torch.sigmoid(out2.squeeze())

        return adv, aux
# For image datasets
X_train = torch.stack(X_train)
X_test = torch.stack(X_test)
for zzz in range(1):
    for t in range(num_tests):
        print(tests[t])
        # Batch size during training
        batch_size = 100


        # Size of z latent vector (i.e. size of generator input)
        nz = 100

        # Size of feature maps in generator
        ngf = 64

        # Size of feature maps in discriminator
        ndf = 64

        # Learning rate for optimizers
        lr = 0.002

        # Beta1 hyperparam for Adam optimizers
        beta1 = 0.5

        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = 1

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                              else "cpu")
        # Number of channels in the training images. For color images this is 3
        nc = 3

        if "classification" in tests[t]:
            out_1 = int(tests[t].split("_")[1])
            out_2 = 1
            length = int(out_1/5)
        elif "regression" in tests[t]:
            out_1 = 1
            out_2 = int(tests[t].split("_")[1])
            length = int(out_2/5)
        elif "double" in tests[t]:
            out_1 = 25
            out_2 = 40
            length_class = 5
            length_reg = 8
        else:
            out_1 = 1
            out_2 = 1
            length = 0


        # Create the generator
        netG = Generator(ngpu).to(device).float()

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netG.apply(weights_init)

        # Print the model

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)

        # Create the Discriminator
        num_classes = 5
        netD = Discriminator(ngpu, out_1, out_2).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(weights_init)

        # Print the model

        noise = torch.randn(batch_size, nc, 128, 128, device=device)
        adv, aux = netD(noise)

        # Initialize BCELoss function
        criterion_adv = nn.BCELoss()
        criterion_aux = nn.MSELoss()

        if (device.type == 'cuda') and (ngpu > 1):
            criterion_adv.to(device)
            criterion_aux.to(device)

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        #checkpoint_dir = '~/ganRegression/content/MNIST_Check/'


        #def create_dir(directory):
        #    """Creates a directory if it does not already exist.
        #    """
        #    if not os.path.exists(directory):
        #        os.makedirs(directory)


        # create_dir(checkpoint_dir)


        #def checkpoint(epoch, G, D, optimizerG, optimizerD, lossG, lossD):
        #    """
        #    Saves the parameters of the generator G and discriminator D.
        #    """
        #    GAN_path = os.path.join(checkpoint_dir, 'GAN.pkl')
        #
        #    torch.save({
        #        'epoch': epoch,
        #        'G_state_dict': G.state_dict(),
        #        'D_state_dict': D.state_dict(),
        #        'optimizerG_state_dict': optimizerG.state_dict(),
        #        'optimizerD_state_dict': optimizerD.state_dict(),
        #        'lossG': lossG,
        #        'lossD': lossD
        #    }, GAN_path)


        #def load_checkpoint(model, checkpoint_name):
        #    model.load_state_dict(torch.load(os.path.join(checkpoint_dir,
        #                                                  checkpoint_name)))


        # Lists to keep track of progress
        img_list = []
        img_list_fixedN = []
        G_losses = []
        D_losses = []
        iters = 0

        sample_interval = 100
        num_epochs = 50

        supervised_batch = batch_size // 2

        netD.train()
        netG.train()
        t0 = time.time()
        a = 16;
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # for epoch in range(1):
            # For each batch in the dataloader
            print(epoch)
            for i in range(int(len(X_train) // batch_size)):
                # Format batch
                # print(X_train.shape)
                real_data = torch.as_tensor(X_train[i * batch_size:i * batch_size + batch_size, :],
                                            dtype=torch.float32).reshape(-1, nc, 128, 128).to(device)
                true_labels = torch.as_tensor(y_train[i * batch_size:i * batch_size + batch_size], dtype=torch.float32)
                b_size = real_data.size(0)
                label_real_adv = torch.full((b_size,), real_label, device=device,
                                            requires_grad=False, dtype=torch.float32)
                label_fake_adv = torch.full((b_size,), fake_label, device=device,
                                            requires_grad=False, dtype=torch.float32)


                netG.zero_grad()
                netD.zero_grad()

                # netG.zero_grad()

                # update generator (g)
                # update supervised discriminator (aux)
                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                x = real_data


                _, auxSup = netD(x)
                if "regression" in tests[t] or "double" in tests[t]:
                    T = conjunction_gen(auxSup[:, 0:length_reg], auxSup[:, 3*length_reg:5*length_reg])
                    S = disjunction_gen(auxSup[:, length_reg:2*length_reg], auxSup[:, 2*length_reg:3*length_reg])

                    I = implication_gen(T, S)
                    A = aggregate_gen(I)
                    auxSup_loss = criterion_aux(A.view(-1), true_labels.to(device).view(-1))
                else:
                    auxSup_loss = criterion_aux(auxSup.view(-1), true_labels.to(device).view(-1))

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                adv_fake, _ = netD(fake.detach())
                advUnsup, _ = netD(real_data)
                if "classification" in tests[t] or "double" in tests[t]:
                    T_fake = conjunction(adv_fake[:, 0:1*length_class], adv_fake[:, 3*length_class:5*length_class])
                    S_fake = disjunction(adv_fake[:, 1*length_class:2*length_class], adv_fake[:, 2*length_class:3*length_class])
                    I_fake = implication(T_fake, S_fake)
                    A_fake = aggregate(I_fake)

                    errD_fake = criterion_adv(A_fake.view(-1), label_fake_adv)

                    T_real = conjunction(advUnsup[:, 0:1*length_class], advUnsup[:, 3*length_class:5*length_class])
                    S_real = disjunction(advUnsup[:, 1*length_class:2*length_class], advUnsup[:, 2*length_class:3*length_class])
                    I_real = implication(T_real, S_real)
                    A_real = aggregate(I_real)

                    errD_real = criterion_adv(A_real, label_real_adv)

                else:
                    errD_fake = criterion_adv(adv_fake.view(-1), label_fake_adv)
                    errD_real = criterion_adv(advUnsup.view(-1), label_real_adv)


                # errD_real = torch.mean(advUnsup)

                errD = 1 / 2 * (errD_real + errD_fake) + auxSup_loss;

                errD.backward()
                # Update D

                optimizerD.step()

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                adv_g, _ = netD(fake)
                if "classification" in tests[t] or "double" in tests[t]:
                    T_g = conjunction(adv_g[:, 0:1*length_class], adv_g[:, 3*length_class:5*length_class])
                    S_g = disjunction(adv_g[:, 1*length_class:2*length_class], adv_g[:, 2*length_class:3*length_class])
                    I_g = implication(T_g, S_g)
                    A_g = aggregate(I_g)
                    errG = criterion_adv(A_g.view(-1), label_real_adv)
                else:
                    errG = criterion_adv(adv_g.view(-1), label_real_adv)

                # errG = errG1;

                errG.backward()

                # Update G
                optimizerG.step()

                iters += 1




        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_valid_scaled = scaler.transform(X_valid)
        #X_test_scaled = scaler.transform(X_test)

        mse_gan_ = []
        mae_gan_ = []
        n_eval_runs = 10

        if "regression" in tests[t] or "double" in tests[t]:
            reg = True
        else:
            reg = False


        def predict(xtest, regression):
            _, ypred = netD(xtest.to(device))
            if regression:
                T = conjunction_gen(ypred[:, 0:1*length_reg], ypred[:, 3*length_reg:5*length_reg])
                S = disjunction_gen(ypred[:, 1*length_reg:2*length_reg], ypred[:, 2*length_reg:3*length_reg])

                I = implication_gen(T, S)

                A = aggregate_gen(I)

                output = A
            else:
                output = ypred

            return output


        def sample(xtest, n_samples):
            y_samples_gan = predict(xtest, reg).cpu().detach().numpy()
            for i in range(n_samples - 1):
                ypred_gan = predict(xtest, reg)
                y_samples_gan = np.hstack([y_samples_gan, ypred_gan.cpu().detach().numpy()])
            median = []
            mean = []
            for j in range(y_samples_gan.shape[0]):
                median.append(np.median(y_samples_gan[j]))
                mean.append(np.mean(y_samples_gan[j]))

            return np.array(mean).reshape(-1, 1), np.array(median).reshape(-1, 1), y_samples_gan

        ypred_gan_test = predict(torch.as_tensor(X_test, dtype=torch.float32).view(-1, nc, 128, 128), reg)
        fig, ax = plt.subplots()
        ax.plot(range(len(ypred_gan_test.cpu().detach().numpy()[:100])),ypred_gan_test.cpu().detach().numpy()[:100].reshape(-1), label="Predicted")
        ax.plot(range(len(y_test[:100])),y_test[:100], label="True")
        ax.legend(loc='upper right')
        if "SGAN" in tests[t]:
            plt.title("Driving Dataset SGAN: True RUL vs. Predicted RUL")
        elif "double" in tests[t]:
            plt.title("Driving Dataset FuzzyGAN Double Injection: True RUL vs. Predicted RUL")
        else:
            plt.title("Driving Dataset FuzzyGAN " + tests[t].split("_")[0] +  " Injection (" + str(int(int(tests[t].split("_")[1])/5)) + " implications): True RUL vs. Predicted RUL")
        plt.ylabel("Angle")
        plt.xlabel("Index")
        if "SGAN" in tests[t]:
            plt.savefig("SGAN_Figures/SGANDriving_new2_" + str(zzz) + ".jpg")
        elif "double" in tests[t]:
            plt.savefig("SGAN_Figures/SGANDriving_FuzzyGANDriving_double_25_40_new2_" + str(zzz) + ".jpg")
        else:
            plt.savefig("SGAN_Figures/SGAN_FuzzyGANDriving_" + tests[t].split("_")[0] + str(int(int(tests[t].split("_")[1])/5)) + "_new2_" + str(zzz) + ".jpg")

        for i in range(n_eval_runs):
            ypred_mean_gan_test_, ypred_median_gan_test_, _ = sample(torch.as_tensor(X_test, dtype=torch.float32).view(-1, nc, 128, 128), 1)
            mae_gan_.append(mean_absolute_error(y_test, ypred_median_gan_test_))
            mse_gan_.append(mean_squared_error(y_test, ypred_mean_gan_test_))

        gan_mae_mean = np.mean(np.asarray(mae_gan_))
        gan_mae_std = np.std(np.asarray(mae_gan_))

        print(f"GAN MAE test: {gan_mae_mean} +- {gan_mae_std}")

        gan_mse_mean = np.mean(np.asarray(mse_gan_))
        gan_mse_std = np.std(np.asarray(mse_gan_))

        print(f"GAN MSE test: {gan_mse_mean} +- {gan_mse_std}")
        if "SGAN" in tests[t]:
            #open text file
            text_file = open("SGAN_outputs/SGAN_Driving_Test" + str(zzz), "w")
        elif "double" in tests[t]:
            text_file = open("SGAN_outputs/SGAN_FuzzyGAN_Driving_double_25_40_Test" + str(zzz), "w")
        else:
            text_file = open("SGAN_outputs/SGAN_FuzzyGAN_Driving_" + tests[t] + "outputs_Test" + str(zzz), "w")


        #write string to file
        n = text_file.write(f"GAN MAE test: {gan_mae_mean} +- {gan_mae_std}\n GAN MSE test: {gan_mse_mean} +- {gan_mse_std}")

        #close file
        text_file.close()
        print(tests[t] + " Finished!")











