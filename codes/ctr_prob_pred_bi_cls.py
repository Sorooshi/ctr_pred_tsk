import os
import gzip
import time
import random
import warnings
import itertools
import numpy as np
import pandas as pd
import ADmetrics as adm
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True, precision=4)

"""
    There are some easier and smarter ways of converting str to int/floating data, e.g.,
    using "eval" method from "literal_eval"; but since this part of code was publicly available
    then I modified it for my own purposed.
    Obviously, some of the features should be encoded using One Hot Encoded technique.
    To this end, any features which I intended to convert is initially assigned as np.dtype(str).
"""

types_train = {
    'id': np.dtype(int),
    'click': np.dtype(int),
    'hour': np.dtype(int),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(str),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str), 
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(str),
    'device_conn_type': np.dtype(str),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21':np.dtype(int)
}

types_test = {
    'id': np.dtype(int),
    'hour': np.dtype(int),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(str),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str), 
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(str),
    'device_conn_type': np.dtype(str),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21':np.dtype(int)
}

"""
There are some serious issues in the data set. To be more precise, 
the data set is a multi-scale data set containing both the categorical 
features and the quantitative features, which is totally fine. 
The correct way of dealing the categorical features is to convert 
them into one-hot encoded vectors. 
Initially, I started to use all features, including the categorical and 
quantitative features so that the categorical ones are converted to 
one-hot encoded vectors. 

Doing so leads me to understand the underlying issue with the dataset. 
Once I converted a categorical feature of the training set my matrix 
size has shape equal to N_1 \times V_1; where N_1 is the number 
of observations and V_1 is the number of columns/features.
Applying the same procedure on the test set leads to a matrix of 
size M \times V_2; where N_2 is the number 
of observations in the test set and V_2 is the number of columns/features. 
if V_1 were equal to V_2 there would not any problems. But since 
this is not the case, we have several options to overcome this issue, 
I have chosen a simple one, that is I limit the number of features 
so that after conversion the aforementioned issue will not occur.
"""

V = ['C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']  # features_to_use

# With selection the above features we indeed eliminate all the categorical features.
# And moreover, some part of the our code, below, will never be used but I leave
# as it was in order to show the correct way of dealing with the data set.


if __name__ == '__main__':

    learning_rate = 1e-5
    batch_size = 64
    n_epochs = 200
    n_classes = 2
    n_units = 100
    tau = 0.85  # user defined threshold to determine how precise the prediction probabilities should be

    # The training set contains over 40 millions of records, due to computational power limits
    # I will randomly sample 1 million of them

    n = 40428967  # total number of records in the clickstream data
    sample_size = 1000000
    skip_values = sorted(random.sample(range(1,n), n-sample_size))
    parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')
    
    """  
    # Once I converted and saved the training data set in order to save time in my future runs
    # Load the train data set
    with gzip.open('../data/train.gz') as f:
        train = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date, dtype=types_train, skiprows = skip_values)
        
    x_in = train.loc[:, train.columns != 'click']
    y_in = train.click.values
    y_in = y_in.reshape(-1, 1)

    x_in.to_csv('../data/x_in.csv')
    np.savetxt('../data/y_in.npy', y_in)
   """

    x_in = pd.read_csv('../data/x_in.csv')
    x_in = x_in.loc[:, x_in.columns != 'id']
    x_in = x_in.loc[:, x_in.columns != 'click']
    
    y_in = np.loadtxt('../data/y_in.npy')

    print('x_in.shape:', x_in.shape)
    print('y_in shape:', y_in.shape)

    y_in = y_in.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_in = one_hot_encoder.fit_transform(y_in)
    y_in = y_in.astype('float32')

    # Load the test data set
#    with gzip.open('../data/test.gz') as f:
#       x_test = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date, dtype=types_test,)
#    x_test.to_csv('../data/x_test.csv')

    x_test = pd.read_csv('../data/x_test.csv')
    idx_path = '../data/sampleSubmission.csv'
    tmp_idx = pd.read_csv(idx_path)

    x_test = x_test.loc[:, x_test.columns != 'id']
    print('x_test.shape:', x_test.shape)
    
    # fixed set of all test samples indices
    fixed_test_idx = set(np.arange(x_test.shape[0]))

    # variable set of all test samples indices (modified at the end of test process)
    variable_test_idx = set(np.arange(x_test.shape[0]))

    # V = set(x_in.columns.tolist())  # set of features

    for v in range(2, len(V) + 1):

        C = list(itertools.combinations(V, v))  # v-th length combination of features

        for c in C:

            print("combination:", len(c), "out of", len(C))

            if len(c) == 1:
                col = ''.join(c).strip().split(',')
            else:
                col = ','.join(c).strip().split(',')
                
            # This section is devoted to convert the categorical features to one-hot encoded vector.
            # And since I limited the features, to be used, to quantitative features,
            # therefore, this part of with the current setting will not be used anymore.
            # (However, I did not remove it for code reviewers to see how I was going to treat
            # the data if the aforementioned issue did not exist)

            x_in_restricted = x_in[col]
            x_test_restricted = x_test[col]

            columns_to_be_encoded = []
            # for col_ in x_in_restricted.columns:
            #    if type(col_) == str:
            #        columns_to_be_encoded.append(col_)

            if len(columns_to_be_encoded) != 0:
                x_in_restricted = pd.get_dummies(data=x_in_restricted, columns=columns_to_be_encoded).values
                x_test_restricted = pd.get_dummies(data=x_test_restricted, columns=columns_to_be_encoded).values
            else:
                x_in_restricted = x_in_restricted.values
                x_test_restricted = x_test_restricted.values

            # splitting train data set into training and validation data set
            x_train_restricted, x_valid_restricted, y_train_restricted, y_valid_restricted = train_test_split(
                x_in_restricted, y_in, test_size=0.2,
                random_state=42, shuffle=True)

            # normalizing the data
            # It is preferable to normalize the data though my computational power does not allow me to do so :(
            # x_train_restricted = x_train_restricted / np.max(x_train_restricted, axis=0)
            # x_valid_restricted = x_valid_restricted / np.max(x_valid_restricted, axis=0)
            # x_test_restricted = x_test_restricted / np.max(x_test_restricted , axis=0)

            print("Shapes:", x_train_restricted.shape, x_valid_restricted.shape, x_test_restricted.shape)

            def create_model():

                model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(n_units, activation='relu', input_shape=(x_train_restricted.shape[1],)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(int(n_units/2), activation='relu',),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(n_classes)
                ])

                model.compile(optimizer='adam',
                              loss=loss_fn,
                              metrics=['accuracy'])

                print(model.summary())

                return model

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=.0001, patience=5, verbose=1, mode='auto',
                baseline=None, restore_best_weights=False)

            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            model = create_model()

            history = model.fit(x=x_train_restricted, y=y_train_restricted,
                                validation_data=(x_valid_restricted, y_valid_restricted),
                                epochs=n_epochs, callbacks=[early_stopping,])

            print("history:", history)

            val_loss, val_acc = model.evaluate(x_valid_restricted, y_valid_restricted, verbose=2)

            acc = history.history['accuracy']

            val_acc = history.history['val_accuracy']

            loss = history.history['loss']

            val_loss = history.history['val_loss']

            model_name = str(v) + ':' + str(v) + 'cmp4v:' + str(c)

            model_path_2_store = '../models/'

            if not os.path.exists(model_path_2_store):
                os.mkdir(model_path_2_store)

            # saving the model for future usage though it is not needed now
            model.save(os.path.join(model_path_2_store, model_name))

            # Make predictions and converting logits to probabilities using Softmax activation
            history_tst = probability_model = tf.keras.Sequential([model,
                                                                   tf.keras.layers.Softmax()
                                                                   ])

            # I was lazy to define separate variables so I copied the same lines of code;
            # and instead I merely replaced the x_val_restricted with x_test_restricted
            y_probs_tst = probability_model.predict(x_valid_restricted)

            print(" ")
            preds_probs = y_probs_tst
            print("preds_probs:", preds_probs.shape)
            print(preds_probs)
            print(" ")
            tmp_preds_labels = np.argmax(preds_probs, axis=1)
            print("tmp_preds_labels:", tmp_preds_labels.shape)
            print(tmp_preds_labels)
            print(" ")

            # It is more preferable to perform such an evaluation on test data set; however, since there is not
            # any ground truth available for test data set, consequently, I evaluated the performance of the
            # proposed algorithm on validation set instead.
            
            name_of_auc_roc_fig = 'roc_auc-' + str(v) + ':' + str(v) + 'cmp4v:' + str(c)

            adm.plot_roc_auv_curve_of_an_algorithm(alg_ms=tmp_preds_labels, gt_ms=y_valid_restricted.argmax(axis=1),
                                                   data_name='ctr', alg_name='fada_bi_cls',
                                                   name_of_auc_roc_fig=name_of_auc_roc_fig, sample_weight=None, case=0)

            name_of_train_val_fig = 'TrainVal-' + str(v) + ':' + str(v) + 'cmp4v:' + str(c)

            # Predicting the labels and the corresponding probabilities on test data set
            y_probs_tst = probability_model.predict(x_test_restricted)

            print(" ")
            preds_probs = y_probs_tst
            print("preds_probs:", preds_probs.shape)
            print(preds_probs)
            print(" ")
            
            preds_labels = np.argmax(preds_probs, axis=1)
            print("preds_labels:", preds_labels.shape)
            print(preds_labels)
            print(" ")
            
            preds_prob_max = np.max(preds_probs, axis=1)
            print("preds_prob_max:", preds_prob_max.shape)
            print(preds_prob_max)
            
            above_thr_preds_prob = preds_prob_max >= tau
            preds_idx = np.arange(preds_prob_max.shape[0])
            above_thr_preds_prob_idx = preds_idx[above_thr_preds_prob]
            print("above_thr_preds:", above_thr_preds_prob_idx)
            
            idx = tmp_idx.iloc[:, 0]

            results_path_2_store = '../results/'

            if not os.path.exists(results_path_2_store):
                os.mkdir(results_path_2_store)
            
            frame = {'id': idx, 'click': preds_prob_max}
            results_2_store = pd.DataFrame(frame)

            results_2_store = pd.DataFrame(
                data=results_2_store, columns=['id', 'click']
            )

            results_2_store.to_csv(
                os.path.join(results_path_2_store,
                             'cls_preds-' + str(v) + ':' + str(v) + 'cmp4v' + str(c) + '.csv'),
                index=False,
            )

            if len(variable_test_idx) == 0:
                print("no more test data!")
                break

        if len(variable_test_idx) == 0:
            print("no more test data!")
            print("len(c):", len(c), c)

            results_2_store.to_csv(os.path.join(results_path_2_store,
                                                'cls_predsF-' + str(v) + ':' + str(v) + 'cmp4v' + str(c) + '.csv'),
                                   index=False, columns=['id', 'click'],
                                   )
            break 



