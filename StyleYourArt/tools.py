# -*- coding: utf-8 -*-
"""
Module contains helper functions.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import PIL
from PIL import Image
import os
from glob import glob
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from itertools import cycle

def print_data_info(df):
    """
    Print information related to dataset.
    """

    ## print header
    print('******************* SUMMARY *******************\n')

    ## print shape
    print("Content")
    print("-----------------------------------------------")
    print("Rows: {:,}".format(df.shape[0]))
    print("Cols: {:,}".format(df.shape[1]))
    print("Number of artists: {:,}".format(df['json_file'].nunique()))
    print("Number of unique styles: {:,}".format(df['style'].nunique()))
    print("Date range: from {:.0f} to {:.0f}".format(df['completion_year'].min(),df['completion_year'].max()))
    print('\n')

    ## info
    print("Info")
    print("-----------------------------------------------")
    print(df.info())


def label(x, label, color):
    """
    Helper function to label the plot in axes coordinates
    """
    ## get axis
    ax = plt.gca()
    ## plot label
    ax.text(0, .2, label, fontweight="bold", color='k',
            ha="left", va="center", transform=ax.transAxes, fontsize=15)


def plot_over_time(df, continuous_feature, class_feature, save=False, palette=sns.color_palette("hls", 25)):
    
    ## set style
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)},font_scale = 1.5)

    ## filter dataframe
    df = df[[continuous_feature, class_feature]]
    
    ## create facegrid
    g = sns.FacetGrid(df,
                    row=class_feature,
                    hue=class_feature,
                    aspect=15,
                    height=1,
                    palette=palette)

    ## draw the densities in a few steps
    g.map(sns.kdeplot,
            continuous_feature,
            clip_on=False,
            shade=True,
            alpha=1,
            lw=1.5,
            bw=.2)
    g.map(sns.kdeplot, continuous_feature, clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    ## add label to each curve
    g.map(label, continuous_feature)

    ## set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.50)

    ## remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    if save:
        plt.savefig("./trend"+class_feature+".png", dpi=150)
    else:
        plt.show()
    sns.set_style("darkgrid")


def display_images(df, styles, dir_feature='file_loc'):
    """
    Display 3 images randomly selected from the input style.
    """
    ## plot
    fig, axes = plt.subplots(len(styles),3, figsize=(16, 6 * len(styles)))
    for r, s in enumerate(styles):
        mask = df['style'] == s
        images = df[mask].sample(3, replace=False)
        locs = images[dir_feature]
        for c, img in enumerate(locs):
            image = Image.open(img)
            data = np.asarray(image)
            axes[r, c].imshow(data)
            axes[r, c].grid(False)
            axes[r, c].set_title(s)
            axes[r, c].set_yticklabels([])
            axes[r, c].set_xticklabels([])
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=False, title=None, cmap=plt.cm.Blues, figsize=(16,16), fontsize=12):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    ## compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    ## only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)
    
    ## qw want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ## rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ## loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize=fontsize,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    ## create file name
    if normalize:
        file_name = save_path+"/"+title.replace(" ", "_")+"_confusion_matrix.png"
    else:
        file_name = save_path+"/"+title.replace(" ", "_")+"_confusion_matrix.png"
    plt.savefig(file_name, dpi=150)


def plot_ROC_curves(y_true, y_pred, save_path, classes, title=None):
    """
    Create ROC plot and save figure
    """

    ## plot linewidth.
    lw = 1
    n_classes = y_pred.shape[1]

    ## compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        style = classes[i]
        fpr[style], tpr[style], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[style] = auc(fpr[style], tpr[style])

    ## compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ## compute macro-average ROC curve and ROC area
    ## first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[classes[i]] for i in range(n_classes)]))

    ## then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        style = classes[i]
        mean_tpr += np.interp(all_fpr, fpr[style], tpr[style])

    ## finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    ## save roc_auc
    pd.Series(roc_auc).to_csv(save_path+"/"+title.replace(" ", "_")+"_AUC_ROC.csv")

    ## plot all ROC curves
    plt.figure(figsize=(8,8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        style = classes[i]
        plt.plot(fpr[style], tpr[style], color='grey', lw=1, alpha=0.3)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path+"/"+title.replace(" ", "_")+"_AUC_ROC.png", dpi=150)


def save_classification_report(y_true, y_pred, save_path, classes, title=None):
    """
    Create classification report and save into txt file
    """
    report = classification_report(y_pred,
                                   y_true,
                                   output_dict=True,
                                   target_names=classes)

    report = pd.DataFrame(report).T
    report.to_csv(save_path+"/"+title.replace(" ", "_")+"_report.csv")


def plot_history(history, save_path):
    """
    Generate three plots:
        1. Loss
        2. Accuracy
        3. Learning Rate
    """
    
    ## create figure
    fig, axes = plt.subplots(1,3,figsize=(16,6))
    history[['loss', 'val_loss']].plot(ax=axes[0], color=['dodgerblue', 'crimson'])
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss - cross-entropy")
    
    history[['accuracy', 'val_accuracy']].plot(ax=axes[1], color=['dodgerblue', 'crimson'])
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Accuracy")
    
    history["lr"].plot(ax=axes[2], color=['green'])
    axes[2].set_xlabel("epoch")
    axes[2].set_title("Learning Rate")
    plt.tight_layout()
    plt.savefig(save_path+"/history.png", dpi=150)


def display_image_counts(image_dir):
    """
    Return the number of images located in each folder stored in image_dir
    """

    if not os.path.exists(image_dir):
        raise Exception("the provided directory does not exist")

    total = 0
    for class_dir in glob(image_dir + "/*/"):
        count = len(os.listdir(class_dir))
        total += count
        print(class_dir, count)

    print("..Total in " + image_dir + ": " + str(total))


if __name__=="__main__":
    display_image_counts("./data/classes/train")
    display_image_counts("./data/classes/test")