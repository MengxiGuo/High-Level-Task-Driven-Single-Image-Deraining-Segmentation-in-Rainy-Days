import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
import random
import matplotlib.image as imgplot
from PIL import Image
import argparse
import cv2
import h5py
import time
from models import DeepLab
from derain_model import PreNet_LSTM, Adversary, LaPulas_Fliter, create_disc_nets, Disc_MultiS_Scale_Loss, \
    create_gen_nets, rain_drop_musk_net
from SSIM import SSIM
from skimage.measure import compare_psnr, compare_ssim

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument("--clean_data_path", default="/home/User1/gmx/derain/datasets/Cityscapes/leftImg8bit/train/")
parser.add_argument("--rain_data_path", default="/home/User1/gmx/derain/datasets/Cityscapes/RainData_train_NoBlur/")
parser.add_argument("--label_path", default="/home/User1/gmx/derain/datasets/Cityscapes/gtFine/train/")
parser.add_argument("--Data_path", default="/home/User1/gmx/derain/datasets/Cityscapes_hfive/")

parser.add_argument("--is_Train", type=bool, default=False)

parser.add_argument("--Test_Real", type=bool, default=False)

parser.add_argument("--Test_multi_weight", type=bool, default=False)

parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--lr", type=float, default=2 * 1e-4)
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument("--milestone", type=int, default=[25, 50, 100], help="When to decay learning rate")

parser.add_argument("--output_stride", type=int, default=16)
# parser.add_argument("--momentum", type=float, default=0.9)
# parser.add_argument("--weight_decay", type=float, default=4 * 1e-5)
parser.add_argument("--backbone", default="resnet")
parser.add_argument("--SS_weight_file", default="./weight/DeepLab_resnet101_clean.pkl")

parser.add_argument("--Drop_musk_weight_file", default="./DR_weight/Rain_drop_musk_5_155.pkl")

parser.add_argument("--Save_path", default="./DR_result/DR_musk/")


derain_transform = transforms.Compose([
    # transforms.RandomSizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225]),
])

SS_transform = transforms.Compose([
    # transforms.RandomSizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def gen_drop_musk(rainy_image, clean_image, gama):
    map = np.abs(rainy_image - clean_image)
    map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    map[map < gama] = 0
    map[map >= gama] = 1
    return map


def batch_tensor(batch_rainy_name_list, batch_clean_name_list, batch_label_name_list, rainy_path, clean_path,
                 label_path, image_size):
    batch_rainy = []
    batch_clean = []
    batch_label = []
    batch_rain_drop_musk = []
    for i in range(len(batch_rainy_name_list)):
        randx = random.randint(0, 1023 - args.image_size)
        randy = random.randint(0, 1919 - args.image_size)
        rainy_image = cv2.imread(rainy_path + batch_rainy_name_list[i].decode())[:, 64:1984, :][
                      randx:randx + image_size,
                      randy:randy + image_size, :]
        # print(rainy_image.shape)
        clean_image = cv2.imread(clean_path + batch_clean_name_list[i].decode())[:, 64:1984, :][
                      randx:randx + image_size,
                      randy:randy + image_size, :]
        label_imge = cv2.imread(label_path + batch_label_name_list[i].decode(), 0)[:, 64:1984][randx:randx + image_size,
                     randy:randy + image_size]

        rainy_drop_musk = gen_drop_musk(rainy_image, clean_image, gama=5)

        rainy_image = np.asarray(derain_transform(Image.fromarray(rainy_image)))
        clean_image = np.asarray(derain_transform(Image.fromarray(clean_image)))
        #rainy_drop_musk = np.asarray(derain_transform(rainy_drop_musk))

        batch_rainy.append(rainy_image)
        batch_clean.append(clean_image)
        batch_rain_drop_musk.append(rainy_drop_musk)

        label_imge[label_imge == 255] = 19
        batch_label.append(label_imge)

    batch_rainy = torch.from_numpy(np.array(batch_rainy)).type(torch.FloatTensor).cuda()
    batch_clean = torch.from_numpy(np.array(batch_clean)).type(torch.FloatTensor).cuda()
    batch_label = torch.from_numpy(np.array(batch_label)).type(torch.LongTensor).cuda()
    batch_rain_drop_musk = torch.from_numpy(np.array(batch_rain_drop_musk)).type(torch.FloatTensor).cuda()
    batch_rain_drop_musk = batch_rain_drop_musk.unsqueeze(1)

    return batch_rainy, batch_clean, batch_label, batch_rain_drop_musk


def SS_output_to_SS_result(SS_output):
    B, C, H, W = SS_output.shape
    SS_result = Variable(torch.zeros(B, H, W)).cuda()
    for i in range(B):
        for h in range(H):
            for w in range(W):
                SS_result[i, h, w] = SS_output[i, :, h, w].argmax()
    return SS_result


def batch_tensor_to_image(batch_tensor):
    return batch_tensor.cpu().detach().numpy().transpose((0, 2, 3, 1))


def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        return data


def Normalize_to_SS(derain_output_tensor):
    B, C, H, W = derain_output_tensor.shape
    norm_derain_output_tensor = (derain_output_tensor - torch.min(derain_output_tensor)) / (
            torch.max(derain_output_tensor) - torch.min(derain_output_tensor))
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    norm_derain_output_tensor = (norm_derain_output_tensor - mean) / std
    return norm_derain_output_tensor


##########################################################
##########################################################
##########################################################
##########################################################
def train_raindrop_musk(args, rain_drop_musk_net):
    Train_clean_image_name = read_data(args.Data_path + "Train_Clean_image_name.h5")
    Train_rain_image_name = read_data(args.Data_path + "Train_Rainy_image_name.h5")
    Train_label_image_name = read_data(args.Data_path + "Train_Label_image_name.h5")

    rainy_name_list = Train_rain_image_name
    clean_name_list = Train_clean_image_name
    label_name_list = Train_label_image_name

    optimizer = torch.optim.Adam(rain_drop_musk_net.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    loss_func = nn.MSELoss().cuda()

    # scheduler1 = MultiStepLR(optimizer_D, milestones=args.milestone, gamma=0.2)
    # scheduler2 = MultiStepLR(optimizer_A, milestones=args.milestone, gamma=0.2)

    counter = 0

    start_time = time.time()
    
    for ep in range(args.epoch):
        # scheduler1.step(ep)
        # scheduler2.step(ep)
        batch_idxs = len(Train_label_image_name) // args.batch_size
        for idx in range(0, batch_idxs):
            # batching......
            batch_rainy_name_list = rainy_name_list[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_clean_name_list = clean_name_list[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_label_name_list = label_name_list[idx * args.batch_size: (idx + 1) * args.batch_size]
            Train_rainy_tensor, Train_clean_tensor, Train_label, Train_musk = batch_tensor(batch_rainy_name_list,
                                                                                           batch_clean_name_list,
                                                                                           batch_label_name_list,
                                                                                           args.rain_data_path,
                                                                                           args.clean_data_path,
                                                                                           args.label_path,
                                                                                           args.image_size)

            counter += 1

            optimizer.zero_grad()

            output_musk = rain_drop_musk_net(Train_rainy_tensor)

            loss = loss_func(output_musk, Train_musk)

            loss.backward()

            optimizer.step()

            ######################################

            if counter % 2 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss1: [%.8f]" \
                      % ((ep + 1), counter, time.time() - start_time, loss))

            if counter % 50 == 0:
                torch.save(rain_drop_musk_net.state_dict(), args.Drop_musk_weight_file)
##################################################################################################################
def cut_batch_tensor(img, sub_number):
    (H, W, C) = img.shape
    batch_tensor_list = []
    with torch.no_grad():
        for i in range(sub_number):
            if (i != sub_number - 1):
                sub_img = img[:, i * (W // sub_number): (i + 1) * (W // sub_number), :]
            else:
                sub_img = img[:, i * (W // sub_number):, :]
            sub_img = derain_transform(Image.fromarray(sub_img)).unsqueeze(0).type(torch.FloatTensor).cuda()
            batch_tensor_list.append(sub_img)
    return batch_tensor_list


def output_list_to_cpu(output_tensor_list):
    for i in range(len(output_tensor_list)):
        # print(output_tensor_list[i].shape)
        output_tensor_list[i] = output_tensor_list[i].squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)) * 255
    return output_tensor_list


def merge_tensor_to_image(output_list):
    merge_img_list = []
    for index in range(len(output_list[0])):
        for i in range(len(output_list)):
            img = output_list[i][index]
            if (i == 0):
                merge_img = img
            else:
                merge_img = np.concatenate((merge_img, img), 1)
        # print(merge_img.shape)
        merge_img_list.append(merge_img)
    return merge_img_list

def test_raindrop_musk(args, rain_drop_musk_net):
    Test_clean_image_name = read_data(args.Data_path + "Test_Clean_image_name.h5")
    Test_rain_image_name = read_data(args.Data_path + "Test_Rainy_image_name.h5")
    # Test_label_image_name = read_data(args.Data_path + "Test_Label_image_name.h5")

    clean_name_list = Test_clean_image_name
    input_name_list = Test_rain_image_name
    # label_name_list = Test_label_image_name

    clean_path = args.clean_data_path
    input_path = args.rain_data_path
    # label_path = args.label_path
    rain_drop_musk_net.load_state_dict(torch.load(args.Drop_musk_weight_file))
    rain_drop_musk_net.eval()
    for index in range(len(input_name_list)):
        input_image = cv2.imread(input_path + input_name_list[index].decode())[:, 64:1984, :]
        clean_image = cv2.imread(clean_path + clean_name_list[index].decode())[:, 64:1984, :]
        rainy_drop_musk = gen_drop_musk(input_image, clean_image, gama=5)
        input_tensor_list = cut_batch_tensor(input_image, 2)
        output_list = []
        for l in range(len(input_tensor_list)):
            output_tensor_list = rain_drop_musk_net(input_tensor_list[l])
            output_tensor_list = [output_tensor_list]
            output_tensor_list = output_list_to_cpu(output_tensor_list)
            output_list.append(output_tensor_list)
        output_image_list = merge_tensor_to_image(output_list)

        for i in range(len(output_image_list)):
            save_path = args.Save_path + "predict/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            musk = output_image_list[i]/255
            print(np.min(musk),np.max(musk))
            
            musk = np.maximum(musk, 0)
            musk /= np.max(musk)
            
            musk = np.uint8(musk*255)
            
            musk = cv2.applyColorMap(musk, cv2.COLORMAP_JET)
            
            rainy_drop_musk = np.uint8(rainy_drop_musk*255)
            rainy_drop_musk = cv2.applyColorMap(rainy_drop_musk, cv2.COLORMAP_JET)
            
            superimposed_img = musk * 0.4 + input_image*0.6
            #cv2.imwrite(save_path + "DR_musk_" + str(index) + "_.png", musk)
            #cv2.imwrite(args.Save_path + "gt/"+ "DR_musk_" + str(index) + "_.png", superimposed_img)
        #cv2.imwrite(args.Save_path + "input/" + str(index) + "_input.png", input_image)
        print("DR_", str(index))
        torch.cuda.empty_cache()


args = parser.parse_args()
Rain_drop_musk_net = rain_drop_musk_net(n_blocks=5).cuda()
Rain_drop_model = nn.DataParallel(Rain_drop_musk_net)
if (args.is_Train):
    train_raindrop_musk(args, Rain_drop_model)
else:
    test_raindrop_musk(args, Rain_drop_model)

