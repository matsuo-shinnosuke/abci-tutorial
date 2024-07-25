import numpy as np
import random
import torch
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def mkdir(path):
    p = ''
    for x in path.split('/'):
        p += x+'/'
        if not os.path.exists(p):
            os.mkdir(p)

def set_loger(path):
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path)
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])

def set_device(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    return device

def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_confusion_matrix(cm, path, title=''):
    cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_acc_curve(history, path):
    plt.plot(history['train_acc'], label='training accuracy', alpha=0.7)
    plt.plot(history['val_acc'], label='validation accuracy', alpha=0.7)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 100)
    plt.grid()
    plt.legend()
    plt.savefig(path+'curve_acc.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_loss_curve(history, path):
    plt.plot(history['train_loss'], label='training loss', alpha=0.7)
    plt.plot(history['val_loss'], label='validation loss', alpha=0.7)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend()
    plt.savefig(f'{path}/curve_loss.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_space(features, labels, path):
    num_classes = max(labels)+1
    show_index = np.random.choice(np.arange(labels.shape[0]), 5000, replace=False)
    features_2d = TSNE(n_components=2, random_state=42).fit_transform(features[show_index])

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(num_classes):
        target = features_2d[labels[show_index] == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=f'class {i}', alpha=0.3, color=f'C{i}')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20, markerscale=2)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.savefig(f'{path}/feature_space.png', bbox_inches='tight', dpi=300)
    plt.close()
