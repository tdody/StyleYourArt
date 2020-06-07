#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module used to train our model and make predictions.
"""

import time, os, re, csv, sys, uuid, joblib, gc, math

try:
    import StyleYourArt.dataloader
    import StyleYourArt.tools
    from StyleYourArt.logger import update_train_log
    import StyleYourArt.lr_finder
except:
    import dataloader
    import tools
    from logger import update_train_log
    import lr_finder

import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from keras.utils.np_utils import to_categorical  
from tensorflow.keras import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = "2.d"
MODEL_VERSION_NOTE = "MobileNetV2+Conv2D"
MODEL_BASE_TAG = "MobileNetV2" ## ResNet50 VGG16 InceptionV3 MobileNetV2 Xception

## model parameters
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

CLASSES_MAPPING = {
    0: 'Abstract Art',
    1: 'Art Nouveau (Modern)',
    2: 'Baroque',
    3: 'Conceptual Art',
    4: 'Cubism',
    5: 'Expressionism',
    6: 'Impressionism',
    7: 'Minimalism',
    8: 'Naïve Art (Primitivism)',
    9: 'Neo-Expressionism',
    10: 'Neoclassicism',
    11: 'Pop Art',
    12: 'Realism',
    13: 'Renaissance',
    14: 'Romanticism',
    15: 'Surrealism',
    16: 'Symbolism',
    17: 'Ukiyo-e'
    }

def create_model(input_shape, num_classes, add_conv2D=True):
    '''
    Returns a keras model.

    Arguments:
        input_shape: tuple containing the shape of the input arrays.
        num_classes: number of classes to be predicted.
    
    Outputs:
        Keras model
    '''
    model = Sequential()
    if add_conv2D:
        model.add(Conv2D(512, 2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Flatten())
    else:
        model.add(Flatten(input_shape=input_shape))

    model.add(Dense(1024, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def get_weight_path(tag):
    '''
    Return path where model weights are stored.

    Arguments:
        tag: model tag (ResNet50 VGG16 InceptionV3 MobileNetV2 Xception)

    Output:
        full path containing the model weights
    '''
    if tag == "ResNet50":
        path = "./models/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    elif tag == "InceptionV3":
        path = "./models/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    elif tag == "MobileNetV2":
        path = "./models/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1_0_224_no_top.h5"
    elif tag == "Xception":
        path = "./models/weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    elif tag == "VGG16":
        path = "./models/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    else:
        path = ""
    return path


def get_base_model(tag, weight_path):
    """
    Create the base model used for transfer learning.

    Arguments:
        tag: string (ResNet50, InceptionV3, MobileNetV2, Xception, VGG16)
        weight_path: location of the base model weights

    Outputs:
        base model with preset weights
    """
    if tag == "ResNet50":
        base_model = ResNet50(
            include_top=False,
            weights=weight_path,
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    elif tag == "InceptionV3":
        base_model = InceptionV3(
            include_top=False,
            weights=weight_path,
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    elif tag == "MobileNetV2":
        base_model = MobileNetV2(
            include_top=False,
            weights=weight_path,
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    elif tag == "Xception":
        base_model = Xception(
            include_top=False,
            weights=weight_path,
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    elif tag == "VGG16":
        base_model = VGG16(
            include_top=False,
            weights=weight_path,
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    else:
        raise Exception("Model tag unknown.")

    return base_model


def organize_directories(df, class_dir, cap_count=None, test_size=0.2, random_state=42, test=False, verbose=True):
    """
    Organize the images into one directory for each class. This is meant to facilitate the use of 
    Keras ImageDataGenerator (flow from directory).

    Arguments:
        df: input dataframe containing the original image data
        class_dir: location to create the training and test directories
        cap_count: limit the number of samples per classes (if None, full set is used)
        test_size: fraction of the dataset to be used for the test set
        random_state: random_state to randomly split the data into training and test set
        test: if True, only 10% of the original images are considered (for debug only)
        verbose: if True, print progress

    Outputs:
        None
    """

    ## check if directory exists
    if cap_count:
        marker = "_" + str(cap_count)
    else:
        marker = ""
    if not os.path.exists(class_dir + marker):
        os.mkdir(class_dir + marker)

    ## create train/test directories
    if not os.path.exists(os.path.join(class_dir + marker, 'train')):
        os.mkdir(os.path.join(class_dir + marker, 'train'))
    if not os.path.exists(os.path.join(class_dir + marker, 'test')):
        os.mkdir(os.path.join(class_dir + marker, 'test'))

    ## TEST ONLY
    ## >>>>>>>>>>>>>>>>>>>>>
    ## split data
    if test:
        df, _ = train_test_split(df, test_size=0.1, random_state=random_state, stratify=df['style'])
    ## <<<<<<<<<<<<<<<<<<<<<

    ## create train and test sets
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['style'])

    ## fill each directory with images
    for name, X in {'train':X_train, 'test':X_test}.items():
        ## create a folder for each class
        for s in X['style'].unique().tolist():
            if not os.path.exists(os.path.join(class_dir + marker, name, s)):
                os.mkdir(os.path.join(class_dir + marker, name, s))

        ## loop over each row of the dataframe and move image based on class
        counter = 0
        if verbose: print("... moving images")
        for index, row in tqdm(X.iterrows(),total=X.shape[0]):
        
            ## check if original image exists
            if not os.path.exists(row['file_loc']):
                continue
            
            ## check if number of items exceeds cap_count
            if cap_count:
                if name == "test":
                    count = int(test_size * cap_count)
                else:
                    count = int(cap_count)
            else:
                count = None
            if count:
                number_files = len(os.listdir(os.path.join(class_dir + marker, name, row['style'])))
                if number_files >= count:
                    continue

            ## create destination path
            destination = os.path.join(class_dir + marker, name, row['style'],str(row['content_id'])+'.png')

            ## read image
            image = Image.open(row['file_loc'])

            ## resize image
            new_image = image.resize((224, 224))

            ## save image
            new_image.convert('RGB').save(destination)
            counter+=1

        if verbose: print("... transfer completed.")
        print("{0}/{1} image found".format(counter, X.shape[0]))


def train_model(data_dir, test=False, from_notebook=False, verbose=True, add_conv2D=False):
    """
    Train model and save weights.

    Arguments:
        data_dir: location of the train and test folders
        test: if test, only use 1 epoch and save results in a special directory
        from_notebook: use to update relative paths
        verbose: print progress and step completions

    External Arguments:
        IMAGE_WIDTH: image width (used for input in base model)
        IMAGE_HEIGHT: image height (used for input in base model)
        BATCH_SIZE: batch size used for training
        MODEL_BASE_TAG: name of the base model (used to retrieve weights and generate bottleneck features)
        EPOCHS: number of epochs

    Outputs:
        checkpoints: folder containing best model weights
        history.csv: history of model training
        history.png: plots of accuracy, loss, and learning rates
        summary.txt: model structure

        Test_Set_AUC_ROC.csv: AUC_ROC for test set
        Test_Set_AUC_ROC.png: AUC_ROC for test set
        Test_Set_confusion_matrix.png: confusion matrix for test set
        Test_Set_Normalize_confusion_matrix.png: normalized confusion matrix for test set
        Test_Set_report.csv: precision,recall,f1-score,support for test set

        Train_Set_AUC_ROC.csv: AUC_ROC for train set
        Train_Set_AUC_ROC.png: AUC_ROC for train set
        Train_Set_confusion_matrix.png: confusion matrix for train set
        Train_Set_Normalize_confusion_matrix.png: normalized confusion matrix for train set
        Train_Set_report.csv: precision,recall,f1-score,support for train set
    """
    ## start timer for runtime
    time_start = time.time()

    ## create location for train and test directory
    if not os.path.exists(data_dir):
        raise Exception("specified data directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'train')):
        raise Exception("training directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'test')):
        raise Exception("specified test directory does not exist.")

    ## adjust relative path
    if from_notebook:
        base_weight_path = '.' + get_weight_path(MODEL_BASE_TAG)
        relative = ".."
    else:
        base_weight_path = get_weight_path(MODEL_BASE_TAG)
        relative = ".."
        
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    ## initialize datagenerator
    datagen = ImageDataGenerator(rescale=1/255.)

    ## run generator
    if verbose: print("...creating generators")
    generator_train = datagen.flow_from_directory(
        train_dir,
        color_mode="rgb",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
        )

    generator_test = datagen.flow_from_directory(
        test_dir,
        color_mode='rgb',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
        )

    ## extract info from generator
    nb_train_samples = len(generator_train.filenames)
    nb_test_samples = len(generator_test.filenames)
    num_classes = len(generator_train.class_indices)
    num_step_train = int(math.ceil(nb_train_samples / BATCH_SIZE))  
    num_step_test = int(math.ceil(nb_test_samples / BATCH_SIZE))  

    ## create path for models if needed
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))

    ## check if bottleneck weights exist
    if "_" in data_dir  :
        model_count = data_dir.split("_")[-1]
        count_tag = "_" + model_count 
    else:
        count_tag = ""
    
    if not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + count_tag + '_bottleneck_features_18_train.npy')):
        if verbose: print("...creating bottleneck features")
        base_model = get_base_model(MODEL_BASE_TAG, base_weight_path)

        ## create bottle neck by passing the training data into the base model
        bottleneck_features_train = base_model.predict_generator(
                                                    generator_train,
                                                    num_step_train,
                                                    verbose=1
                                                    )
        ## save bottleneck weights
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + count_tag + '_bottleneck_features_18_train.npy'), bottleneck_features_train)
        del bottleneck_features_train

        ## create bottle neck by passing the training data into the base model
        bottleneck_features_test = base_model.predict_generator(
                                                    generator_test,
                                                    num_step_test,
                                                    verbose=1
                                                    )
        ## save bottleneck weights
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + count_tag + '_bottleneck_features_18_test.npy'), bottleneck_features_test)
        del bottleneck_features_test
        gc.collect()
 
    ## load the bottleneck features saved earlier  
    if verbose: print("...loading bottleneck features")
    train_data = np.load(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + count_tag + '_bottleneck_features_18_train.npy'))
    test_data = np.load(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + count_tag + '_bottleneck_features_18_test.npy'))

    ## get the class labels for the training data, in the original order  
    train_labels = generator_train.classes
    test_labels = generator_test.classes

    ## shuffle 
    np.random.seed(42)
    orders = np.arange(train_data.shape[0])
    np.random.shuffle(orders)

    train_data = train_data[orders]
    train_labels = train_labels[orders]

    if verbose: print("   ...training data shape", train_data.shape)
    if verbose: print("   ...testing data shape", train_data.shape)
    if verbose: print("   ...training label data shape", train_labels.shape)
    if verbose: print("   ...testing label data shape", train_labels.shape)

    ## convert the training labels to categorical vectors
    if verbose: print("...encoding labels")
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    ## compute class weights
    if verbose: print("...computing class weights")
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_labels.argmax(axis=1)),
                                                      train_labels.argmax(axis=1))
    
    ## build additional layers
    if verbose: print("...creating model")
    model = create_model(input_shape=train_data.shape[1:], num_classes=num_classes, add_conv2D=add_conv2D)

    ## save info about models
    model_name = re.sub("\.","_",str(MODEL_VERSION))

    ## TEST ONLY
    ## >>>>>>>>>>>>>>>>>>>>>
    if test:
        model_name+="_test"
    ## <<<<<<<<<<<<<<<<<<<<<

    ## create directory for version specific
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))
    if not os.path.exists(os.path.join(relative,"models",model_name)):
        os.mkdir(os.path.join(relative,"models",model_name))
    if not os.path.exists(os.path.join(relative,"models",model_name,"checkpoints")):
        os.mkdir(os.path.join(relative,"models",model_name,"checkpoints"))

    ## OPTIMIZER
    #optimizer = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999)
    optimizer = optimizers.SGD(lr=0.02, clipnorm=1.)

    ## compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=["accuracy"]
        )

    ## CALLBACKS
    ## progress
    if from_notebook:
        callbacks = [TQDMNotebookCallback(leave_outer=True, leave_inner=True)]
        model_verbose = 0
    else:
        callbacks = []
        model_verbose = 2

    ## reduce LR
    lrate = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor = 0.4,
        patience=3,
        verbose=1,
        min_lr = 0.000001
    )
    callbacks.append(lrate)

    ## early stopping
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=10,
        min_delta=0.005
        )
    callbacks.append(es)

    ## save
    checkpoints = ModelCheckpoint(
        os.path.join(relative,"models",model_name,"checkpoints",model_name+".h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        period=1
    )
    callbacks.append(checkpoints)

    ## compute step size and epoch count
    n_epochs = EPOCHS

    ## adjust parameters for test
    if test:
        n_epochs = 1

    if verbose: print("...training model")
    history = model.fit(train_data,
                        train_labels,
                        epochs=n_epochs,
                        batch_size=BATCH_SIZE, 
                        verbose=model_verbose,
                        validation_data=(test_data, test_labels),
                        callbacks=callbacks,
                        class_weight=dict(enumerate(class_weights)))

    ## save model summary
    if verbose: print("...saving model summary")
    with open(os.path.join(relative,"models",model_name,"summary.txt"),'w') as fh:
        # pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    ## save history into a csv file
    if verbose: print("...saving training history")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(relative,"models",model_name,"history.csv"))

    ## create plots for learning rate, accuracy, and loss
    save_path = os.path.join(relative,"models",model_name)
    plot_history(history_df, save_path) 

    ## save model
    if verbose: print("...saving model weights")
    model.save(os.path.join(relative,"models",model_name,'my_model_18.h5'))

    ## make prediction on train and test set
    if verbose: print("...making predictions on train and test sets")
    y_pred_train_proba = model.predict(train_data, verbose=1)
    y_pred_test_proba = model.predict(test_data, verbose=1)

    ## compute running time
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    if verbose: print("...update training log")
    update_train_log(
        MODEL_VERSION,
        MODEL_BASE_TAG,
        MODEL_VERSION_NOTE,
        train_data.shape,
        test_data.shape,
        {"accuracy": accuracy_score(train_labels.argmax(axis=1), y_pred_train_proba.argmax(axis=1))},
        {"accuracy": accuracy_score(test_labels.argmax(axis=1), y_pred_test_proba.argmax(axis=1))},
        EPOCHS,
        BATCH_SIZE,
        [str(type(callback)) for callback in callbacks],
        runtime,
        optimizer.get_config(),
        test,
        from_notebook
    )

    ## save confusion matrix
    if verbose: print("...creating confusion matrix plot")
    y_true = train_labels.argmax(axis=1)
    y_pred = y_pred_train_proba.argmax(axis=1)
    ##classes = np.array(list(generator_train.class_indices.keys()))
    ## special class order
    classes = np.array([
        'Abstract Art', 'Art Nouveau (Modern)', 'Baroque', 'Conceptual Art',
        'Cubism', 'Expressionism', 'Impressionism',
        'Minimalism', 'Naïve Art (Primitivism)', 'Neo-Expressionism',
        'Neoclassicism', 'Pop Art', 'Realism',
        'Renaissance', 'Romanticism', 'Surrealism',
        'Symbolism', 'Ukiyo-e'
        ])

    plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=False, title="Training Set", cmap=plt.cm.Blues, figsize=(16,16))
    plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=True, title="Training Set Normalize", cmap=plt.cm.Blues, figsize=(16,16))
    y_true = test_labels.argmax(axis=1)
    y_pred = y_pred_test_proba.argmax(axis=1)
    plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=False, title="Test Set", cmap=plt.cm.Oranges, figsize=(16,16))
    plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=True, title="Test Set Normalize", cmap=plt.cm.Oranges, figsize=(16,16))

    ## save AUC (one vs. rest) curves
    if verbose: print("...creating ROC-AUC plot")
    plot_ROC_curves(train_labels, y_pred_train_proba, save_path, classes, title="Train Set")
    plot_ROC_curves(test_labels, y_pred_test_proba, save_path, classes, title="Test Set")

    ## save classification report
    if verbose: print("...saving classification report")
    save_classification_report(train_labels.argmax(axis=1), y_pred_train_proba.argmax(axis=1), save_path, classes, title="Train Set")
    save_classification_report(test_labels.argmax(axis=1), y_pred_test_proba.argmax(axis=1), save_path, classes, title="Test Set")


def find_best_lr(data_dir, optimizer_name, base_model_tag, min_lr=1e-5, max_lr=1e-2, from_notebook=False, verbose=True):
    """
    
    """

    ## create location for train and test directory
    if not os.path.exists(data_dir):
        raise Exception("specified data directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'train')):
        raise Exception("training directory does not exist.")

    ## adjust relative path
    if from_notebook:
        relative = ".."
    else:
        relative = ".."
        
    train_dir = os.path.join(data_dir, 'train')

    ## initialize datagenerator
    datagen = ImageDataGenerator(rescale=1/255.)

    ## run generator
    generator_train = datagen.flow_from_directory(
        train_dir,
        color_mode="rgb",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
        )

    ## extract info from generator
    nb_train_samples = len(generator_train.filenames)
    num_classes = len(generator_train.class_indices)
    num_step_train = int(math.ceil(nb_train_samples / BATCH_SIZE))  

    ## create path for models if needed
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))

    ## check if bottleneck weights exist
    if "_" in data_dir  :
        model_count = data_dir.split("_")[-1]
        count_tag = "_" + model_count 
    else:
        count_tag = ""
 
    ## load the bottleneck features saved earlier  
    train_data = np.load(os.path.join(relative, 'models', 'bottlenecks', base_model_tag + count_tag + '_bottleneck_features_18_train.npy'))
    
    ## get the class labels for the training data, in the original order  
    train_labels = generator_train.classes
    
    ## convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    ## compute class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_labels.argmax(axis=1)),
                                                      train_labels.argmax(axis=1))
    
    ## build additional layers
    model = create_model(input_shape=train_data.shape[1:], num_classes=num_classes)

    ## save info about models
    model_name = re.sub("\.","_",str(MODEL_VERSION))

    ## create directory for version specific
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))
    if not os.path.exists(os.path.join(relative,"models",model_name)):
        os.mkdir(os.path.join(relative,"models",model_name))
    if not os.path.exists(os.path.join(relative,"models",model_name,"checkpoints")):
        os.mkdir(os.path.join(relative,"models",model_name,"checkpoints"))

    ## OPTIMIZER
    if optimizer_name=="Adam":
        optimizer = optimizers.Adam(beta_1=0.9, beta_2=0.999)
    elif optimizer_name=="SGD":
        optimizer = optimizers.SGD(clipnorm=1.)
    else:
        raise Exception("Please choose an optimizer Adam or SGD")

    ## compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=["accuracy"]
        )

    ## CALLBACKS
    n_epochs = 2
    lr_finder = LRFinder(optimizer_name, base_model_tag, min_lr=min_lr, max_lr=max_lr,steps_per_epoch=int(train_data.shape[0]/BATCH_SIZE), epochs=n_epochs)
    history = model.fit(train_data,
                        train_labels,
                        epochs=n_epochs,
                        batch_size=BATCH_SIZE, 
                        verbose=1,
                        callbacks=[lr_finder],
                        class_weight=dict(enumerate(class_weights)))

    ## save model summary
    ##if verbose: print("...saving model summary")
    ##with open(os.path.join(relative,"models",model_name,"summary.txt"),'w') as fh:
    ##    # pass the file handle in as a lambda function to make it callable
    ##    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    return lr_finder


def load_my_models(version, tag, model_dir=None, from_notebook=False):
    """    
    Load models (base and top).

    Arguments:
        version: model version as string
        tag: string (ResNet50, InceptionV3, MobileNetV2, Xception, VGG16)
        from_notebook: use True to offset relative paths
    """
    ## adjust relative path
    weight_path = get_weight_path(tag)
    if from_notebook:
        relative = ".."
        weight_path = ".." + weight_path
    else:
        relative = "."

    ## if data path not specified, use generic
    if not model_dir:
        model_dir = os.path.join(relative,"models", version.replace(".", "_"))

    ## check if model exists
    if not os.path.isfile(os.path.join(model_dir, "my_model_18.h5")):
        raise Exception("No model is specified directory {}".format(model_dir))
    
    ## check if weights exist
    if not os.path.isfile(weight_path):
        raise Exception("No base model weights in specified directory {}".format(weight_path))

    ## load model
    top_model = models.load_model(os.path.join(model_dir, "my_model_18.h5"))
    #>>>>>>>>>>>graph_top = get_default_graph(tag, )

    ## load base model
    base_model = get_base_model(tag, weight_path)
    #>>>>>>>>>>>graph_base = get_default_graph()

    #>>>>>>>>>>>return base_model, top_model, top_graph
    return base_model, top_model


def make_prediction(base_model, top_model, image, filename):
    """

    """
    ## resize image
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)).convert('RGB')
    image = np.asarray(image)

    ## pre-process
    image = image / 255
    image = np.expand_dims(image, axis=0)

    ## create bottleneck encoding
    bottleneck_prediction = base_model.predict(image)

    ## classify image
    predictions = top_model.predict(bottleneck_prediction)

    ## sort classes
    orders = predictions.argsort()[0]

    ## create prediction
    predicted_class = CLASSES_MAPPING[predictions.argmax(axis=1)[0]]
    
    ## create plot
    fig, ax = plt.subplots(figsize=(8,8))
    ax.barh(np.array(list(CLASSES_MAPPING.values()))[orders],predictions[0][orders], color="darkorange")
    ax.set_title("Prediction: " + predicted_class)
    for i, v in enumerate(predictions[0][orders]):
        if v>0.005:
            ax.text(v + 0.01, i - .2, "{:.0f}%".format(100*v), color='k', fontsize=15)
    ax.set_xlim(0, predictions[0].max()+0.15)
    ax.axes.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig("./app/static/prediction/prediction_{}.png".format(filename), dpi=150)
    return predicted_class

if __name__ == "__main__":

    train_model('../data/classes', test=False, from_notebook=False, verbose=True, add_conv2D=True)