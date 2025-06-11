import time
from tqdm import tqdm
from configs import bcolors
from utils import *

def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]


def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x, target_size) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x, target_size) for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)


def load_train_data(path='./training/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + line.split(',')[5])
            ys.append(float(line.split(',')[6]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading training data, running time: {} seconds, size {}'.format(
        time.time() - start_load_time, len(train_xs)) + bcolors.ENDC)
    return train_generator, len(train_xs)

def load_additional_train(path=None, batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    img_paths = [img for img in os.listdir(path) if img.endswith(".png")]

    for img_path in tqdm(img_paths):
        xs.append(path + '/' + img_path)
        steer = [float(i[1:-1]) for i in img_path.split('.png')[0].split('_')[1:4]]
        steer.sort()
        ys.append(steer[1])
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, (xs, ys)

def load_val_data(path=None, batch_size=64, shape=(100, 100), start=None, end=None):
    xs = []
    ys = []
    start_load_time = time.time()
    gen_img_folder = path+'/center'
    img_paths = ['center/'+img for img in os.listdir(gen_img_folder) if img.endswith(".jpg")][start:end]

    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            filename = line.split(',')[5]
            if filename in img_paths:
                xs.append(path + filename)
                ys.append(float(line.split(',')[6]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading validation data, running time: {} seconds, size {}'.format(
        time.time() - start_load_time, len(train_xs)) + bcolors.ENDC)
    return train_generator, len(train_xs)


def load_augmented_test_data(add_xs, add_ys, path=None, batch_size=64, shape=(100, 100), start=None, end=None):
    xs = []
    ys = []
    start_load_time = time.time()
    gen_img_folder = path+'/center'
    img_paths = ['center/'+img for img in os.listdir(gen_img_folder) if img.endswith(".jpg")][start:end]

    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            filename = line.split(',')[5]
            if filename in img_paths:
                xs.append(path + filename)
                ys.append(float(line.split(',')[6]))
    # shuffle list of images
    for i in add_xs:
        xs.append(i)
    for i in add_ys:
        ys.append(i)
    import pdb;
    pdb.set_trace()
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    import pdb; pdb.set_trace()
    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)

def load_test_data(path='./testing/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    print('CH2_final_evaluation.csv contains gt labels')
    with open(path + 'CH2_final_evaluation.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading testing data, running time: {} seconds, size {}'.format(
        time.time() - start_load_time, len(train_xs)) + bcolors.ENDC)
    return train_generator, len(train_xs)

if __name__ == '__main__':
    gen_img_folder = '/home/jzhang2297/anomaly/CH2_001/center'
    img_paths = ['center/' + img for img in os.listdir(gen_img_folder) if img.endswith(".jpg")]
    for i in tqdm(img_paths):
        print(i)
        preprocess('/home/jzhang2297/anomaly/CH2_001/' + i, (100,100))
