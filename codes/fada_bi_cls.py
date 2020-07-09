import os
import time
import warnings
import itertools
import numpy as np
import ADmetrics as adm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    learning_rate = 1e-4
    batch_size = 100
    n_epochs = 20
    n_classes = 2
    n_units = 80
    tau = 0.7

    # Generate synthetic data / load data sets
    x_in, y_in = make_classification(n_samples=1000, n_features=10, n_informative=4, n_redundant=2,
                                     n_repeated=2, n_classes=2, n_clusters_per_class=2, weights=[0.5, 0.5],
                                     flip_y=0.01, class_sep=1.0, hypercube=True,
                                     shift=0.0, scale=1.0, shuffle=True, random_state=42)

    x_in = x_in.astype('float32')
    x_in = x_in / x_in.max(axis=0)  # normalizing the data
    y_in = y_in.astype('float32').reshape(-1, 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    y_in = one_hot_encoder.fit_transform(y_in)
    y_in = y_in.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.4, random_state=42, shuffle=True)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, shuffle=True)
    print("shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

    # fixed set of all test samples indices
    fixed_test_idx = set(np.arange(x_test.shape[0]))

    # variable set of all test samples indices (modified at the end of test process)
    variable_test_idx = set(np.arange(x_test.shape[0]))

    V = x_train.shape[1]
    V = set(range(V))

    ind = 0

    for v in range(1, len(V) + 1):

        C = list(itertools.combinations(V, v))

        for c in C:

            ind += 1

            print("combination:", len(c), "out of", len(C))

            if len(c) == 1:
                z_dim = 1  # latent_dim
                o_dim = 1  # original_dim
                i_dim = int(2 * z_dim)  # intermediate_dim

            elif len(c) >= 2:
                o_dim = len(c)  # original_dim
                z_dim = int(len(c) / 2)  # latent_dim
                i_dim = int(2 * z_dim)  # intermediate_dim

            x_train_restricted = x_train[:, c]
            print(x_train_restricted.shape)
            y_train_restricted = y_train[:, ]
            x_valid_restricted = x_val[:, c]
            y_valid_restricted = y_val[:, ]
            x_test_restricted = x_test[:, c]
            y_test_restricted = y_test[:, ]

            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(n_units, activation='relu', input_shape=(o_dim,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(n_classes)
            ])

            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            model.compile(optimizer='adam',
                          loss=loss_fn,
                          metrics=['accuracy'])

            print(model.summary())

            history = model.fit(x=x_train_restricted, y=y_train_restricted,
                                validation_data=(x_valid_restricted, y_valid_restricted),
                                epochs=n_epochs)

            print("history:", history)

            val_loss, val_acc = model.evaluate(x_valid_restricted, y_valid_restricted, verbose=2)

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']


            model.save('../models/')

            # Make predictions and converting logits to probabilities using Softmax activation
            history_tst = probability_model = tf.keras.Sequential([model,
                                                     tf.keras.layers.Softmax()])

            y_probs_tst = probability_model.predict(x_test_restricted)

            print(" ")
            # preds_probs = tf.subtract(y_probs_tst, log_q_z_x_tst).numpy()
            preds_probs = y_probs_tst
            print("preds_probs:", preds_probs.shape)
            print(preds_probs)
            print(" ")
            tmp_preds_labels = np.argmax(preds_probs, axis=1)
            print("tmp_preds_labels:", tmp_preds_labels.shape)
            print(tmp_preds_labels)
            print(" ")
            preds_prob_max = np.max(preds_probs, axis=1)
            print("preds_prob_max:", preds_prob_max.shape)
            print(preds_prob_max)
            above_thr_preds_prob = preds_prob_max >= tau
            preds_idx = np.arange(preds_prob_max.shape[0])
            above_thr_preds_prob_idx = preds_idx[above_thr_preds_prob]
            print("above_thr_preds:", above_thr_preds_prob_idx)

            variable_test_idx = variable_test_idx.difference(set(above_thr_preds_prob_idx))

            name_of_auc_roc_fig = 'roc_auc-' + str(v) + ':' + str(v) + 'cmb4v:' + str(c)

            adm.plot_roc_auv_curve_of_an_algorithm(alg_ms=tmp_preds_labels, gt_ms=y_test_restricted.argmax(axis=1),
                                                   data_name='synthetic', alg_name='fada_bi_cls',
                                                   name_of_auc_roc_fig=name_of_auc_roc_fig, sample_weight=None, case=0)

            name_of_train_val_fig = 'TrainVal-' + str(v) + ':' + str(v) + 'cmb4v:' + str(c)

            adm.plot_train_valid_curves_of_algorithm(acc=acc, val_acc=val_acc, v=v, C=C,
                                                     loss=loss, val_loss=val_loss, n_epochs=n_epochs,
                                                     data_name='synthetic',
                                                     name_of_train_val_fig=name_of_train_val_fig,
                                                     )

            if len(variable_test_idx) == 0:
                print("no more test data!")
                break

        if len(variable_test_idx) == 0:
            print("no more test data!")
            print("len(c):", len(c), c)

    print("ind:", ind)



