import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import scikitplot as skplt
import torch
import numpy as np
from torch.utils.data.dataloader import pin_memory_batch
from torch.utils.data.dataloader import default_collate
#from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')


__author__ = 'jeven'
__email__ = 'jeven.z@aliyun.com'


def most_common(x, dim=1):
    """ Find most common element along dim, x should be 2D tensor """
    results = []
    for i in range(x.size()[0]):
        ind = torch.cuda.LongTensor([i])
        candidates = torch.Tensor.squeeze(x.index_select(0, ind))
        candidates = candidates.cpu().numpy()
        winner = np.bincount(candidates).argmax()
        results.append(int(winner))
    return torch.cuda.LongTensor(results)


def plot_roc(y_true, y_pred, epoch, exp_name, classnum=4):
    # print(y_true.shape, y_pred.shape)
    idx = y_pred.astype(int)
    y_prob = np.zeros((len(y_pred), classnum))
    y_prob[np.arange(len(y_pred)), idx] = 1.0

    skplt.metrics.plot_roc(y_true, y_prob)

    target_dir = os.path.join('models', exp_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    imgpath = os.path.join(target_dir, 'roc_' + str(epoch) + '.png')
    print('saving roc figure to: ', imgpath)
    plt.savefig(imgpath, format='png', dpi=900)
    # plt.show()
    pass


def compute_saliency_maps(X, y, model, classnum):
    """
    使用模型图像(image)X和标记(label)y计算正确类的saliency map.

    Inputs:
    - X: Tensor of shape (N, 3, H, W)
    - y: LongTensor of shape (N,)
    - model: pretrained model

    Return:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Wrap the input tensors in Variables
    X_var = torch.autograd.Variable(X, requires_grad=True)
    y_var = torch.autograd.Variable(y)
    if torch.cuda.is_available():
        X_var = X_var.cuda()
        y_var = y_var.cuda()
    saliency = None
    scores = model(X_var)

    # 得到正确类的分数，scores为[classnum]的Tensor
    scores = scores.gather(1, y_var.view(-1, 1)).squeeze()

    scores.backward(torch.FloatStorage(
        [1.0] * classnum).cuda())  # 参数为对应长度的梯度初始化
#     scores.backward() 必须有参数，因为此时的scores为非标量，为5个元素的向量

    saliency = X_var.grad.data

    saliency = saliency.abs()
    saliency, i = torch.argmax(saliency)  # take max channel of RGB
    saliency = saliency.squeeze()  # 去除1维
#     print(saliency)

    return saliency


def show_saliency_maps(model, X, y, exp_name, classnum):
    # Convert X and y from numpy arrays to Torch Tensors
    # X_tensor = torch.cat([preprocess(Image.fromarray(x.cpu().numpy())) for x in X], dim=0)
    # X_tensor = X
    # y_tensor = y
    print(X.size(), y.size())
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X, y, model, classnum)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]

    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(transforms.ToPILImage(X[i]))
        plt.axis('off')
        plt.title(exp_name[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)

    target_dir = os.path.join('models', exp_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    imgpath = os.path.join(target_dir, 'saliency_' + str(y.data[0]) + '.png')
    print('saving saliency figure to: ', imgpath)
    plt.savefig(imgpath, format='png', dpi=900)
    # plt.show()


def plot_training_hist(history, exp_name):
    target_dir = os.path.join('models', exp_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    imgpath = os.path.join(target_dir, 'learning_acc.png')
    print('saving accuracy figure to: ', imgpath)
    plt.savefig(imgpath, format='png', dpi=900)
    # plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    imgpath = os.path.join(target_dir, 'learning_loss.png')
    print('saving loss figure to: ', imgpath)
    plt.savefig(imgpath, format='png', dpi=900)
    # plt.show()


'''
class CustomDataLoaderIter(DataLoaderIter):
    def __init__(self, loader):
        super(CustomDataLoaderIter, self).__init__(loader)
        self.num_workers = 0
        self.indices = None

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            if self.indices is None:
                # may raise StopIteration
                self.indices = next(self.sample_iter)
            batch = self.collate_fn([self.dataset[i] for i in self.indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)

            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    def cur_batch(self):
        return self.__next__()

    def reset_indices(self):
        self.indices = None


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False):
        super(CustomDataLoader, self).__init__(dataset, batch_size, shuffle,
                                               sampler, batch_sampler, num_workers,
                                               collate_fn, pin_memory,
                                               drop_last)

    def __iter__(self):
        self.iter = CustomDataLoaderIter(self)
        return self.iter

    def get_cur_batch(self):
        return self.iter.cur_batch()[0]


'''
