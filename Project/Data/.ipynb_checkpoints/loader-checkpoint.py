import os
import glob
from PIL import Image
import numpy as np
import time


dir_path   = os.path.dirname(os.path.realpath(__file__))

train_path = './fruits-360_dataset/fruits-360/Training'
test_path = './fruits-360_dataset/fruits-360/Test'


def _load_data(path, label_map=None):
    """
    :param path: sees all the folders in 'path' as labels to be assigned to images in respective folders
    directory structure should look like this: path/[label]/[images_for_that_label]
    :param label_map:  label map (optional) tells which number to be assigned to which numeric labels
    must be a dict. label_map[name_of_label] = unique_number_for_label
    :return: data_array, numeric_labels_array, label_map
    use of label map only is useful in case there are labels not present in testing data, for example.
    e.g.
    > train_x, train_y, label_map = load_data('./dataset/train')
    now use the same map for data
    > test_x, test_y, _ = load_data('./dataset/test', label_map)
    """

    labels = os.listdir(path)
    labels.sort()

    # pre-empt the size for faster arrays.
    all_files = glob.glob(path + '/*/*.jpg')
    n = len(all_files)

    images = np.zeros((n, 100, 100, 3), dtype=np.uint8)
    img_labels = np.zeros(n, dtype=np.uint8)

    # data_labels = []
    if label_map:
        user_map = True
    else:
        user_map = False
        label_map = {}

    label_i = 0
    img_i = 0
    sum_arr = np.zeros((100,100,3), dtype=np.float64)
    sq_arr = np.zeros((100,100,3), dtype=np.float64)
    for label in labels:
        if label[0] == '.':
            # ignore folders beginning with 'dot' e.g. '.DS_Store' in Mac OS
            continue

        if not user_map:
            label_map[label] = label_i
            label_i += 1
        img_paths = os.listdir(path + '/' + label)

        for img_fname in img_paths:
            img_arr = np.array(Image.open('{}/{}/{}'.format(path, label, img_fname)))
            # img_arr = imread('{}/{}/{}'.format(path, label, img_fname))
            # volumes.append(img_arr)
            images[img_i, :, :, :] = img_arr
            float_arr = img_arr / 255
            sum_arr += float_arr
            sq_arr += float_arr ** 2
            img_labels[img_i] = label_map[label]
            img_i += 1
            # img_labels.append(label_map[label])

    mean = sum_arr / img_i
    mean_sq = sq_arr / img_i
    var = mean_sq - (mean ** 2)
    std = np.sqrt(var)
    return images, img_labels, label_map, mean, std


def load_dataset(train_path, test_path):
    if not os.path.exists('dataset.npz'):
        print("Reading all files. This may take a couple of minutes. Please wait.")
        t0 = time.time()
        train_x, train_y, label_map, train_mean, train_std = _load_data(train_path)
        t1 = time.time()
        print("{:.2f} loaded training data".format(t1 - t0))
        test_x, test_y, _, test_mean, test_std = _load_data(test_path, label_map)
        t2 = time.time()
        print("{:.2f} loaded testing data".format(t2 - t1))
        # train_mean = np.mean(train_x, axis=0)
        # train_std = np.std(train_x, axis=0)
        #
        # test_mean = np.mean(test_x, axis=0)
        # test_std = np.mean(test_x, axis=0)
        t3 = time.time()
        print("{:.2f} calculated statistics for data".format(t3 - t2))
        np.savez('dataset.npz',
                 train_x=train_x,
                 train_y=train_y,
                 test_x=test_x,
                 test_y=test_y,
                 train_mean=train_mean,
                 train_std=train_std,
                 test_mean=test_mean,
                 test_std=test_std)
        t4 = time.time()
        print("{:.2f} saved as a single file 'dataset.npz' for quick access".format(t4 - t3))
    else:
        print("Existing file 'dataset.npz' found.")
        t0 = time.time()
        ds = np.load('dataset.npz')
        train_x = ds['train_x']
        train_y = ds['train_y']
        test_x = ds['test_x']
        test_y = ds['test_y']
        train_mean = ds['train_mean']
        train_std = ds['train_std']
        test_mean = ds['test_mean']
        test_std = ds['test_std']
        t1 = time.time()
        print("{:.2f} Loaded data.".format(t1 - t0))

    return train_x, train_y, test_x, test_y, train_mean, train_std, test_mean, test_std


    

def main():
    train_x, train_y, \
        test_x, test_y, \
        train_mean, train_std, \
        test_mean, test_std = load_dataset(train_path, test_path)

if __name__ == '__main__':
    main()