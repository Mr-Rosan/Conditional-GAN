import jittor as jt
import argparse
import numpy as np
import math
import os
import pickle
from jittor import nn
from jittor import init
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
from PIL import Image

if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='This is a implementation of a simple conditional GAN using jittor')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate using adam')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--input_vector_size', type=int, default=100, help='random vector size of Generator input')
parser.add_argument('--embedding_size', type=int, default=10, help='embedding vector size')
parser.add_argument('--image_size', type=int, default=32, help='square image size')
parser.add_argument('--channel', type=int, default=1, help='channel size')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
parser.add_argument('--output_sample', type=bool, default=True,
                    help='Whether to output sample according to sample_interval')

cmd_parse = parser.parse_args()

image_size = cmd_parse.image_size
channel = cmd_parse.channel
embedding_size = cmd_parse.embedding_size
image_shape = (channel, image_size, image_size)
batch_size = cmd_parse.batch_size
learning_rate = cmd_parse.learning_rate
b1 = cmd_parse.b1
b2 = cmd_parse.b2
epochs = cmd_parse.epochs
sample_interval = cmd_parse.sample_interval
random_input_vector_size = cmd_parse.input_vector_size
output_sample = cmd_parse.output_sample

transform = transform.Compose([
    transform.Resize(image_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])


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

def get_data():
    return MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)


# change matrix data into image format and save it
def save_image(img, path, n_row=10, padding=5):
    # get parameter:N denotes batch_size,C denotes channel,W denotes width,H denotes height
    N, C, W, H = img.shape
    # N must be divide by n_row so it can generate a proper matrix
    if N % n_row != 0:
        print("N%n_row!=0")
        return
    # get column
    n_col = int(N / n_row)
    # to save all image
    img_all = []
    # get image at matrix(i,j)
    for i in range(n_col):
        # img_ save the images which are in the same row
        img_ = []
        for j in range(n_row):
            # get the image at (i,j)
            img_.append(img[i * n_row + j])
            # get the initial image with zero padding
            img_.append(np.zeros((C, W, padding)))
        # np.concatenate() is used to concatenate two matrix
        # add images in the same row to img_all
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C, padding, img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C, padding, img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C, img.shape[1], padding)), img], 2)
    min_ = img.min()
    max_ = img.max()
    img = (img - min_) / (max_ - min_) * 255
    img = img.transpose((1, 2, 0))
    if C == 3:
        img = img[:, :, ::-1]
    elif C == 1:
        img = img[:, :, 0]
    Image.fromarray(np.uint8(img)).save(path)

if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    #adversarial_loss = nn.MSELoss()

    #os.makedirs("images", exist_ok=True)
    #os.makedirs("saved_models", exist_ok=True)

    #data_loader = get_data()

    #loss_g, loss_d = train(data_loader, generator, discriminator, adversarial_loss)

    #g_loss_file = open('g_loss_list.pkl', 'wb')
    #d_loss_file = open('d_loss_list.pkl', 'wb')
    #pickle.dump(loss_g, g_loss_file)
    #pickle.dump(loss_d, d_loss_file)
    #g_loss_file.close()
    #d_loss_file.close()

    
    discriminator.load_parameters(pickle.load(open('saved_models/discriminator_last.pkl', 'rb')))
    d_loader=MNIST(train=False, transform=transform).set_attrs(batch_size=1, shuffle=False)
    total=0
    right=0
    for i, (img, label) in enumerate(d_loader):
        # real_img denotes image from MNIST, and label denotes the corresponding label from MNIST
        real_img = jt.array(img)
        label = jt.array(label)
        # get result from Discriminator by sending it real image and it's label
        valid_real = discriminator(real_img, label)
        total+=1
        if valid_real>0.5:
            right+=1
    print(right/total)
#python3.7 -m pip install jittor==1.3.1.53