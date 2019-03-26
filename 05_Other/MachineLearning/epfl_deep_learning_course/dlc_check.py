#!/usr/bin/env python

import torchvision
import dlc_practical_prologue as prologue

train_input, _, _, _ = prologue.load_data(cifar = False)

images = train_input.narrow(0, 0, 64).view(-1, 1, 28, 28)
images /= images.max()

print('Writing check-mnist.png')
torchvision.utils.save_image(images, 'check-mnist.png')

train_input, _, _, _ = prologue.load_data(cifar = True)

images = train_input.narrow(0, 0, 64).view(-1, 3, 32, 32)
images /= images.max()

print('Writing check-cifar.png')
torchvision.utils.save_image(images, 'check-cifar.png')
