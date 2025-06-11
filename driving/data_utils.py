import time
from utils import *


def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]

def data_generator_img(xs, ys, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            X = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            gen_state = 0
        else:
            X = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            gen_state += batch_size
        yield np.array(X), np.array(y)

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

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)


def load_test_data(path='./testing/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'final_example.csv', 'r') as f:
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

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)


def DaveDataset(batch_size=64, no_generator=False):
    xs = []
    ys = []
    # 45406 images
    with open("/home/jzhang2297/data/dave_test/driving_dataset/data.txt") as f:
        for line in tqdm(f):
            xs.append("/home/jzhang2297/data/dave_test/driving_dataset/" + line.split()[0])
            # the paper by Nvidia uses the inverse of the turning radius,
            # but steering wheel angle is proportional to the inverse of turning radius
            # so the steering wheel angle in radians is used as the output
            # *-1 to be consistent w/ udacity (pos=left, neg=right)
            steering_ratio = 15  # or use 16, 18 depending on car type
            steering_angle_wheel_deg = float(line.split()[1]) / steering_ratio
            steering_angle = -1 * float(steering_angle_wheel_deg) * 3.14159265 / 180
            ys.append(steering_angle)

    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_xs = xs[:int(len(xs) * 0.6)]
    train_ys = ys[:int(len(xs) * 0.6)]

    val_xs = xs[-int(len(xs) * 0.4):-int(len(xs) * 0.2)]
    val_ys = ys[-int(len(xs) * 0.4):-int(len(xs) * 0.2)]

    test_xs = xs[-int(len(xs) * 0.2):]
    test_ys = ys[-int(len(xs) * 0.2):]
    print('number of training imgs', len(train_xs))
    print('number of validation imgs', len(val_xs))
    print('number of test imgs', len(test_xs))

    train_generator = data_generator(train_xs, train_ys,
                                     batch_size=batch_size)

    if no_generator:
        return train_xs, train_ys, val_xs, val_ys, test_xs, test_ys, np.array(steer)
    return train_generator, val_xs, val_ys, test_xs, test_ys, np.array(steer)