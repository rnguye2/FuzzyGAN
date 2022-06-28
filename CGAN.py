import importlib
import matplotlib.pyplot as plt
import dataset, metrics, plotting, config, network
from models import cgan_model
import numpy as np
import tensorflow as tf
import random

importlib.reload(network)
importlib.reload(dataset)
importlib.reload(metrics)
importlib.reload(plotting)
importlib.reload(config)
importlib.reload(cgan_model)


import os

dataset_config = config.DatasetConfig(scenario="pumadyn")

assert(dataset_config.scenario == "CA-housing"
      or dataset_config.scenario == "ailerons"
      or dataset_config.scenario == "CA-housing-single"
      or dataset_config.scenario == "comp-activ"
      or dataset_config.scenario == "pumadyn"
      or dataset_config.scenario == "bank"
      or dataset_config.scenario == "abalone"
      or dataset_config.scenario == "census-house"
      or dataset_config.scenario == "driving"
      or dataset_config.scenario == "age")
fig_dir = f"../FuzzyGAN/figures/{dataset_config.scenario}"

try:
    os.mkdir(fig_dir)
    print(f"Directory {fig_dir} created ")
except FileExistsError:
    print(f"Directory {fig_dir} already exists replacing files in this notebook")

random_seed = 1985

if dataset_config.scenario == "CA-housing" or dataset_config.scenario == "CA-housing-single":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.0001, lr_disc=0.001, dec_gen=0, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "ailerons":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.0001, lr_disc=0.0005, dec_gen=0, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "comp-activ":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.005, lr_disc=0.001, dec_gen=0.0001, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "pumadyn":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.001, lr_disc=0.001, dec_gen=0.001, dec_disc=0.001,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "bank":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.001, lr_disc=0.001, dec_gen=0.001, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "abalone":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.001, lr_disc=0.001, dec_gen=0.001, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "census-house":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.001, lr_disc=0.001, dec_gen=0.0001, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "driving":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.001, lr_disc=0.001, dec_gen=0.0001, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=100, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "age":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.001, lr_disc=0.001, dec_gen=0.0001, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=100, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
)



# Set random seed
np.random.seed(exp_config.model.random_seed)
random.seed(exp_config.model.random_seed)

import tensorflow
tensorflow.random.set_seed(exp_config.model.random_seed)

X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(scenario=exp_config.dataset.scenario,
                                                                    seed=exp_config.model.random_seed)

X_train.shape, len(X_valid), len(X_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#X_train_scaled = X_train
#X_valid_scaled = X_valid
#X_test_scaled = X_test
print(np.shape(X_train))

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Comment for pumadyn
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#y_test = min_max_scaler.fit_transform(y_test.reshape(-1,1)).reshape(-1)
#y_train = min_max_scaler.transform(y_train.reshape(-1,1)).reshape(-1)

#y_train = y_train.reshape(-1)
#y_test = y_test.reshape(-1)

print(np.max(y_train))
print(np.min(y_train))

print(np.max(y_test))
print(np.min(y_test))


#import xgboost as xgb
#print("Starting xg_reg")
#xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1,eta=0.3, learning_rate = 0.1,
#                max_depth = 5, alpha = 10, n_estimators = 2000)
#xg_reg.fit(X_train_scaled,y_train)
#print("Finished training xg_reg")
#from sklearn.metrics import mean_absolute_error

#ypred_xg_test = xg_reg.predict(X_test_scaled)
#xg_mae = mean_absolute_error(ypred_xg_test, y_test)
#print(xg_mae)

#cov_xg = np.mean((y_test - ypred_xg_test)**2)
#print(metrics.gaussian_NLPD(y_test, ypred_xg_test, np.ones(len(ypred_xg_test)) * cov_xg, "XG"))


num_outs = [25, 35]
for j in range(len(num_outs)):
    cgan = cgan_model.CGAN(exp_config, num_outs[j])

    from pickle import load
    #scaler = load(open('scaler.pkl', 'rb'))

    d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true = cgan.train(X_train_scaled, y_train,
                                                                                    epochs=exp_config.training.n_epochs,
                                                                                    batch_size=exp_config.training.batch_size)
    ypred_gan_test = cgan.predict(tf.convert_to_tensor(X_test_scaled))
    fig, ax = plt.subplots()
    ax.plot(range(len(ypred_gan_test[:100])),ypred_gan_test[:100].reshape(-1), label="Predicted")
    ax.plot(range(len(y_test[:100])),y_test[:100], label="True")
    ax.legend(loc='upper right')
    plt.title("Pumadyn Dataset FuzzyGAN Regression Injection ( " + str(num_outs[j])  + "outputs): True RUL vs. Predicted RUL")
    plt.ylabel("Rate")
    plt.xlabel("Index")
    plt.savefig("FuzzyGANPumadyn_Regression_" + str(num_outs[j]) + "outs_new.jpg")


    #plotting.plots(d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true, fig_dir, exp_config.run.save_fig)


    ypred_mean_gan_test, ypred_median_gan_test, ypred_gan_sample_test = cgan.sample(tf.convert_to_tensor(X_test_scaled),
                                                                                exp_config.training.n_samples)

    ypred_mean_gan_train, ypred_median_gan_train, ypred_gan_sample_train = cgan.sample(tf.convert_to_tensor(X_train_scaled),
                                                                                exp_config.training.n_samples)


    import keras

    dropout_rate = 0.1

    # Comparable architecture to GAN

    # Original: 32,5,2
    #           32,5,2
    #           16,5,(2,3)
    #           2,5,(2,3)
    # GAN: 100,75,75,75,75,50,1
    #model = keras.models.Sequential([
    #    keras.layers.Conv2D(32, 5, strides=2, activation="relu", input_shape=X_train_scaled.shape[1:]),
    #    keras.layers.Conv2D(32, 5, strides=2, activation="relu"),
        #keras.layers.Conv2D(32, 3, strides=2, activation="relu"),
    #    keras.layers.Conv2D(16, 5, strides=(2,3), activation="relu"),
    #    keras.layers.Conv2D(2, 5, strides=(2,3), activation="relu"),
    #    keras.layers.Flatten(),
    #    keras.layers.Dense(500, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(500, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(500, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(100, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(100, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(50, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(1, activation="linear"),
    #])

    #model = keras.models.Sequential([
    #    keras.layers.Conv1D(4, 5, strides=2, activation="relu", input_shape=X_train_scaled.shape[1:]),
    #    keras.layers.MaxPooling1D(2),
    #    keras.layers.Conv1D(2, 5, strides=2, activation="relu"),
    #    keras.layers.MaxPooling1D(2),
    #    keras.layers.Conv1D(2, 5, strides=2, activation="relu"),
    #    keras.layers.MaxPooling1D(2),
    #    keras.layers.Flatten(),
    #    keras.layers.Dense(500, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(500, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(500, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(100, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(100, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(50, activation="relu"),
    #    keras.layers.Dropout(dropout_rate),
    #    keras.layers.Dense(1, activation="linear"),
    #])
    #model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=0.001, decay=0.1))
    #
    #callbacks = [keras.callbacks.EarlyStopping(patience=10)]
    #history = model.fit(X_train_scaled, y_train,
    #                    validation_data=(X_valid_scaled, y_valid), epochs=1,
    #                    callbacks=callbacks)

    #seed = network.seed
    #random_normal = keras.initializers.RandomNormal(seed=exp_config.model.random_seed)
    #model = keras.models.Sequential([keras.layers.LSTM(64, input_shape=(80*256, 4)),
    #        keras.layers.BatchNormalization(),
    #        keras.layers.Dense(1, activation="linear", kernel_initializer=random_normal)])
    #model = Model(inputs=x, outputs=lstm_output)
    #model.compile(loss="mean_squared_error", optimizer=tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.1))

    #history = model.fit(
    #    X_train_scaled.reshape(-1,80*256,4), y_train, validation_data=(X_valid_scaled.reshape(-1,80*256,4), y_valid), batch_size=64, epochs=100
    #)

    #plotting.plot_learning_curves(history)

    #ypred_nn_test = model.predict(X_test_scaled.reshape(-1,80*256,4))
    #fig2, ax2 = plt.subplots()
    #ax2.plot(range(len(ypred_nn_test)),ypred_nn_test.reshape(-1), label="Predicted")
    #ax2.plot(range(len(y_test)),y_test.reshape(-1), label="True")
    #ax2.legend(loc='upper right')
    #plt.title("Bearing Dataset LSTM (100 samples): True RUL vs. Predicted RUL")
    #plt.ylabel("RUL")
    #plt.xlabel("Index")
    #plt.savefig("LSTMBearing100.jpg")


    #import GPy

    run_hyperopt_search = True
    rbf = True

    #if rbf:
    #    kernel = GPy.kern.RBF(input_dim=cgan.x_input_size, variance=variance, lengthscale=length)
    #else:
    #    kernel = GPy.kern.sde_RatQuad(input_dim=X_train_scaled.shape[1], variance=variance, lengthscale=length, power=power)

    #gpr = GPy.models.GPClassification(X_train_scaled, y_train.reshape(-1, 1), kernel, noise_var=noise_var)

    #if run_hyperopt_search:
    #    gpr.optimize(messages=True)


    # plotting.plot_densities_joint(y_test, ypred_nn_test, ypred_mean_gan_test, ypred_gp_test,
    #                              "Linear-vs-GAN-vs-GP P(y) density", fig_dir=fig_dir,
    #                              prefix="all_marginalized", save_fig=exp_config.run.save_fig, at_x=True)

    #plotting.plot_datadistrib_joint(y_test, ypred_nn_test, ypred_mean_gan_test, ypred_gp_test,
    #                                "Linear-vs-GAN-vs-GP P(y) density", fig_dir=fig_dir,
    #                                prefix="all_marginalized", save_fig=exp_config.run.save_fig)

    n_eval_runs = 10

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from pickle import load

    mse_gan_= []
    mae_gan_ = []

    for i in range(n_eval_runs):
        ypred_mean_gan_test_, ypred_median_gan_test_, _ = cgan.sample((X_test_scaled), exp_config.training.n_samples)
        mae_gan_.append(mean_absolute_error(y_test, ypred_median_gan_test_))
        mse_gan_.append(mean_squared_error(y_test, ypred_mean_gan_test_))
        #score_gan_.append(score(y_test, ypred_mean_gan_test_))

    #nn_mae = mean_absolute_error(ypred_nn_test, y_test)
    #gp_mae = mean_absolute_error(ypred_gp_test, y_test)
    gan_mae_mean = np.mean(np.asarray(mae_gan_))
    gan_mae_std = np.std(np.asarray(mae_gan_))
    print("Nonsense NN results")
    #print(f"NN MAE test: {nn_mae}")
    #print(f"GP MAE test: {gp_mae}")
    print(f"GAN MAE test: {gan_mae_mean} +- {gan_mae_std}")


    #nn_mse = mean_squared_error(ypred_nn_test, y_test)

    #gp_mse = mean_squared_error(ypred_gp_test, y_test)
    gan_mse_mean = np.mean(np.asarray(mse_gan_))
    gan_mse_std = np.std(np.asarray(mse_gan_))
    #mdn_mse_mean = np.mean(np.asarray(mse_mdn_))
    #mdn_mse_std = np.std(np.asarray(mse_mdn_))

    #print(f"NN MSE test: {nn_mse}")
    #print(f"GP MSE test: {gp_mse}")
    print(f"GAN MSE test: {gan_mse_mean} +- {gan_mse_std}")
    #print(f"MDN MSE test: {mdn_mse_mean} +- {mdn_mse_std}")


    #nn_r2 = r2_score(ypred_nn_test, y_test)

    #gp_mse = mean_squared_error(ypred_gp_test, y_test)
    #gan_score_mean = np.mean(np.asarray(score_gan_))
    #gan_score_std = np.std(np.asarray(score_gan_))
    #mdn_mse_mean = np.mean(np.asarray(mse_mdn_))
    #mdn_mse_std = np.std(np.asarray(mse_mdn_))

    #print(f"NN R2 Score test: {nn_r2}")
    #print(f"GP MSE test: {gp_mse}")
    #print(f"GAN RScore test: {gan_score_mean} +- {gan_score_std}")
    #print(f"MDN MSE test: {mdn_mse_mean} +- {mdn_mse_std}")



    #cov_nn = np.mean((y_test - ypred_nn_test)**2)
    #nn_nlpd = metrics.gaussian_NLPD(y_test, ypred_nn_test, np.ones(len(ypred_nn_test)) * cov_nn, "NN")

    #gp_nlpd = metrics.gaussian_NLPD(y_test, ypred_gp_test, cov_test, "GP")

    #gan_nlpd_train, w, lls = metrics.Parzen(cgan, X_valid_scaled, y_valid, n_sample=exp_config.training.n_samples)
    #nlpd_ = []
    #for i in range(n_eval_runs):
    #    nlpd_.append(metrics.Parzen_test(cgan, X_test_scaled, y_test, w, exp_config.training.n_samples))
    #gan_nlpd_test = np.mean(nlpd_)
    #gan_nlpd_std_test = np.std(nlpd_)

    #print(f"GAN Train NLLH: {gan_nlpd_train}")
    #print(f"GAN Test NLLH: mean {gan_nlpd_test} std {gan_nlpd_std_test}")

    with open('FuzzyGAN_Pumadyn_Regression_' + str(num_outs[j]) + 'outs_Test1.txt', 'w') as f:
        f.write(f"===Test MAE===\n")
        #file.write(f"NN MAE test: {nn_mae}\n")
        #file.write(f"GP MAE test: {gp_mae}\n")
        f.write(f"GAN MAE test: {gan_mae_mean} +- {gan_mae_std}\n")
        f.write(f"===Test MSE===\n")
        #file.write(f"NN MSE test: {nn_mse}\n")
        #file.write(f"GP MSE test: {gp_mse}\n")
        f.write(f"GAN MSE test: {gan_mse_mean} +- {gan_mse_std}\n")
        #file.write(f"===Test R2 Score===\n")
        #file.write(f"NN R2 Score test: {nn_r2}\n")
        # file.write(f"GP MSE test: {gp_mse}\n")
        #file.write(f"GAN R2 Score test: {gan_score_mean} +- {gan_score_std}\n")
        #file.write(f"===Test NLPD===\n")
        #file.write(f"NN Gaussian NLPD: {nn_nlpd}\n")
        #file.write(f"GP Gaussian NLPD: {gp_nlpd}\n")
        #file.write(f"GAN NLPD: {gan_nlpd_test} +- {gan_nlpd_std_test}\n")
        f.close()

