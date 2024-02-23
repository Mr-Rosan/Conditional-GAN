import jittor as jt
import argparse
import os
import numpy as np
import math
from jittor import nn
import pickle
from jittor.dataset.mnist import MNIST
import jittor.transform as transform

parser = argparse.ArgumentParser()
parser.add_argument('--input_vector_size', type=int, default=100, help='random vector size of Generator input')
parser.add_argument('--embedding_size', type=int, default=10, help='embedding vector size')
parser.add_argument('--image_size', type=int, default=32, help='square image size')
parser.add_argument('--channel', type=int, default=1, help='channel size')
cmd_parse = parser.parse_args()
print(cmd_parse)

image_shape = (cmd_parse.channel, cmd_parse.image_size, cmd_parse.image_size)

image_size = cmd_parse.image_size
channel = cmd_parse.channel
embedding_size = cmd_parse.embedding_size
random_input_vector_size = cmd_parse.input_vector_size

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(embedding_size, embedding_size)

        self.model = nn.Sequential()

        self.model.append(nn.Linear(random_input_vector_size + embedding_size, 128))
        self.model.append(nn.LeakyReLU(0.2))

        self.model.append(nn.Linear(128, 256))
        self.model.append(nn.LeakyReLU(0.2))

        self.model.append(nn.Linear(256, 512))
        self.model.append(nn.LeakyReLU(0.2))

        self.model.append(nn.Linear(512, 1024))
        self.model.append(nn.LeakyReLU(0.2))

        self.model.append(nn.Linear(1024, int(np.prod(image_shape))))
        self.model.append(nn.Tanh())

    def execute(self, noise, y):
        input_data = jt.contrib.concat((self.label_embedding(y), noise), dim=1)
        img = self.model(input_data)
        img = img.view((img.shape[0], *image_shape))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(embedding_size, embedding_size)

        self.model = nn.Sequential()

        self.model.append(nn.Linear(embedding_size + int(np.prod(image_shape)), 512))
        self.model.append(nn.LeakyReLU(0.2))

        self.model.append(nn.Linear(512, 512))
        self.model.append(nn.Dropout(0.4))
        self.model.append(nn.LeakyReLU(0.2))
 
        self.model.append(nn.Linear(512, 1))

    def execute(self, img, y):
        input_data = jt.contrib.concat((img.view((img.shape[0], (-1))), self.label_embedding(y)), dim=1)
        d_result = self.model(input_data)
        return d_result

import cv2
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
        img_all.append(np.concatenate(img_, 2))
    img = np.concatenate(img_all, 1)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

# ----------
#  Test
# ----------
def get_data():
    return MNIST(train=False, transform=transform).set_attrs(batch_size=1, shuffle=False)

def number_test():
    number = "17582805429"
    n_row = len(number)
    adversarial_loss = nn.MSELoss()
    z = jt.array(np.random.normal(0, 1, (n_row, random_input_vector_size))).float32().stop_grad()
    labels = jt.array(np.array([int(number[num])for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z,labels)
    print(n_row)
    save_image(gen_imgs.numpy(), "result.png", nrow=n_row)

if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    generator.load_parameters(pickle.load(open('saved_models/generator_last.pkl', 'rb')))
    discriminator.load_parameters(pickle.load(open('saved_models/discriminator_last.pkl', 'rb')))
    number_test()