from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import Generator, Discriminator
import torchvision.utils as vutils
from noisetransformation import AddGaussianNoise
import time
import matplotlib.animation as animation
from IPython.display import HTML
import pickle
from torchvision.utils import save_image
import image_slicer
import os
import shutil
import random

if not os.path.exists("images_lpips"):
    os.makedirs("images_lpips")
if not os.path.exists("Real_images"):
    os.makedirs("Real_images")
if not os.path.exists("Fake_images"):
    os.makedirs("Fake_images")

data_transforms = {
    'train': transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor(),
                                 transforms.Normalize((.5, .5, .5), (.5, .5, .5))])  # to [-1, 1] range
}

noise_addition = transforms.Compose([AddGaussianNoise(0., 0.1)])

lsun = datasets.LSUN('data/living_room_train_lmdb', ['living_room_train'], transform=data_transforms['train'])

sub1 = list(range(0, len(lsun), 10))
sub2 = list(range(5, len(lsun), 10))
sub3 = list(range(8, len(lsun), 10))
subset1 = torch.utils.data.Subset(lsun, sub1)
subset2 = torch.utils.data.Subset(lsun, sub2)
subset3 = torch.utils.data.Subset(lsun, sub3)

dir_path = os.path.dirname(os.path.realpath(__file__))


# Show me a nice living room
def show_image(dataLoader):
    for idx, (image, target) in enumerate(dataLoader):
        plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        print(idx, target)
        plt.show()
        break
    pass


# Use CUDA if possible
def to_cuda(item):
    if torch.cuda.is_available():
        return item.cuda()
    return item


# weights should be gaussian intialized
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


# Tensor with labels for real
def reals(size, real_label):
    data = torch.full((size, ), real_label)
    data = data.cuda()
    return data


# Tensor with labels for fake
def fakes(size, fake_label):
    data = torch.full((size, ), fake_label)
    data = data.cuda()
    return data


# Create noise
def noise(size, latent):
    n = torch.randn(size, latent, 1, 1) # play with this as well
    return n.cuda()


# start full training
def train(eps=30, bs=8, glr=0.0002, dlr=0.0002, real_label=0.7, fake_label=0.3):
    torch.cuda.empty_cache()
    latent_size = 128
    noiseAdd = True
    print(torch.cuda.get_device_name(0))
    print(f'Starting training epochs: {eps}, batch size: {bs}, learning rate g: {glr} d: {dlr}, noise addition: {noiseAdd}'
          f', labels: {real_label}, {fake_label}')
    # Initialize

    G_losses = []
    D_losses = []
    img_list = []
    num_test_samples = 64
    test_noise = noise(num_test_samples, latent_size)
    generator = Generator()
    # generator.load_state_dict(torch.load("models\\generator_128_9_0.0002.pth")) # comment out while training
    discriminator = Discriminator()
    # discriminator.load_state_dict(torch.load("models\\discriminator_128_9_0.0002.pth")) # comment out while training
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    generator = generator.cuda()
    discriminator = discriminator.cuda()

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=dlr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=glr, betas=(0.5, 0.999))

    loss = torch.nn.BCELoss()
    subset = 2
    random_labels = False # noisy labels
    if subset == 1:
        dataLoaderTrain = DataLoader(subset1, batch_size=bs, shuffle=False)
    elif subset == 2:
        dataLoaderTrain = DataLoader(subset2, batch_size=bs, shuffle=False)

    for ep in range(eps):
        start = time.time()
        for idx, (real_data, target) in enumerate(dataLoaderTrain):

            discriminator.zero_grad()
            real_data = real_data.cuda()
            if noiseAdd:
                real_data = noise_addition(real_data)
            pred_real = discriminator(real_data).view(-1)
            if random_labels:
                errorD_real = loss(pred_real, reals(real_data.size(0), random.uniform(
                    real_label, real_label + 2 * (1 - real_label))))
            else:
                errorD_real = loss(pred_real, reals(real_data.size(0), real_label))
            errorD_real.backward()
            D_x = pred_real.mean().item()

            fake_data = generator(noise(real_data.size(0), latent_size))
            fake_data = fake_data.cuda()
            if noiseAdd:
                fake_data = noise_addition(fake_data.cuda())
            pred_fake = discriminator(fake_data.detach()).view(-1)
            if random_labels:
                errorD_fake = loss(pred_fake, fakes(real_data.size(0), random.uniform(0, fake_label)))
            else:
                errorD_fake = loss(pred_fake, fakes(real_data.size(0), fake_label))
            errorD_fake.backward()
            errD = errorD_fake + errorD_real
            D_G_z1 = pred_fake.mean().item()
            d_optimizer.step()

            generator.zero_grad()
            pred = discriminator(fake_data).view(-1)
            errG = loss(pred, reals(real_data.size(0), real_label))
            errG.backward()
            D_G_z2 = pred.mean().item()
            g_optimizer.step()

            if idx % 50 == 0 and idx > 1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                ep, eps, idx, len(dataLoaderTrain), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                G_losses.append(errG.item())
                D_losses.append(errD.item())

            if (idx % 2000 == 0) or ((ep == eps - 1) and (idx == len(dataLoaderTrain) - 1)):
                with torch.no_grad():
                    fake = generator(test_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        if ep == eps - 1:

            model = f'noise-{noiseAdd}_noisylabels-{random_labels}_{bs}_{ep + 1}_{glr}-{dlr}_subset-{subset}_labels_{real_label}-{fake_label}'
            import os
            if not os.path.exists(f'models/{model}'):
                os.makedirs(f'models/{model}')

            with open(f'models/{model}/g-losses.pickle', 'wb+') as handle:
                pickle.dump(G_losses, handle)

            with open(f'models/{model}/d-losses.pickle', 'wb+') as handle:
                pickle.dump(D_losses, handle)

            torch.save(generator.state_dict(), f'models/{model}/generator.pth')
            torch.save(discriminator.state_dict(), f'models/{model}/discriminator.pth')

            plt.figure(figsize=(10,5))
            plt.title(f"Generator and Discriminator Loss for batch size: {bs}, learning rate: g: {glr} d: {dlr} and {eps} epochs")
            plt.plot(G_losses,label="G")
            plt.plot(D_losses,label="D")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            # plt.show()
            plt.savefig(f'models/{model}/losscape.png')

            # fig = plt.figure(figsize=(8,8))
            # plt.axis("off")
            # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
            # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
            # ani.save(f'plots/vid_bs{bs}_lr{lr}_ep{ep}.mp4')

            # Grab a batch of real images from the dataloader


            # # Plot the real images
            # plt.figure(figsize=(15,15))
            # plt.subplot(1,2,1)
            # plt.axis("off")
            # plt.title("Real Images")
            # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].cuda()[:64], padding=5, normalize=True).cpu(),(1,2,0)))
            #
            # # Plot the fake images from the last epoch
            # plt.subplot(1,2,2)
            # plt.axis("off")
            # plt.title("Fake Images")
            # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            # plt.show()

            path = f'models/{model}'
            txt = f'{model}.txt'

            if os.path.exists(path):
                shutil.rmtree(path)

            if not os.path.exists(f'{path}/fake_images'):
                os.makedirs(f'{path}/fake_images')

            if not os.path.exists(f'{path}/images_lpips'):
                os.makedirs(f'{path}/images_lpips')

            if not os.path.exists(f'{path}/real_images'):
                os.makedirs(f'{path}/real_images')

            with open(f"{path}/{txt}", "w+"):
                pass

            save_image(fake, f'{path}/fake_images//1.png')
            save_image(fake, f'{path}/images_lpips//1.png')

            dataLoaderTrain_eval = DataLoader(subset3, batch_size=64, shuffle=False)
            real_batch = next(iter(dataLoaderTrain_eval))
            for idx, i in enumerate(real_batch[0][:64]):
                save_image(i, f'{path}/real_images//F_{idx}.png')
                save_image(i, f'{path}/images_lpips//F_{idx}.png')

            image_slicer.slice(f'{path}/fake_images/1.png', 64)
            image_slicer.slice(f'{path}/images_lpips/1.png', 64)
            os.remove(f"{path}/fake_images//1.png")
            os.remove(f"{path}/images_lpips//1.png")

            os.system(f"python fid_score.py {path}/real_images {path}/fake_images --gpu 0")
            os.system(f"python compute_dists_pair.py -d {path}/images_lpips -o {path}/{txt} --all-pairs --use_gpu")


if __name__ == "__main__":
    for i, j, k, l, m in zip([16],
                             [0.7],
                             [0.3],
                             [0.0002],
                             [0.0003]):
        train(eps=30, bs=i, glr=l, dlr=m, real_label=j, fake_label=k)

