import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.vgg import vgg16
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torch.nn.functional as F
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
    create_gen_nets, rain_drop_musk_net, create_refine_nets, create_coarse_nets
from discriminator import Discriminator
from SSIM import SSIM
from skimage.measure import compare_psnr, compare_ssim

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument("--clean_data_path", default="/home/User1/gmx/derain/datasets/Cityscapes/leftImg8bit/train/")
parser.add_argument("--rain_data_path", default="/home/User1/gmx/derain/datasets/Cityscapes/RainData_train_NoBlur/")
parser.add_argument("--label_path", default="/home/User1/gmx/derain/datasets/Cityscapes/gtFine/train/")
parser.add_argument("--Data_path", default="/home/User1/gmx/derain/datasets/Cityscapes_hfive/")

parser.add_argument("--is_Train", type=bool, default=True)

parser.add_argument("--Test_Real", type=bool, default=False)

parser.add_argument("--Test_multi_weight", type=bool, default=True)

parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--lr", type=float, default=2 * 1e-4)
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument("--milestone", type=int, default=[199, 250], help="When to decay learning rate")

parser.add_argument("--output_stride", type=int, default=16)
# parser.add_argument("--momentum", type=float, default=0.9)
# parser.add_argument("--weight_decay", type=float, default=4 * 1e-5)
parser.add_argument("--backbone", default="resnet")
parser.add_argument("--SS_weight_file", default="./weight/DeepLab_resnet101_clean.pkl")

parser.add_argument("--DR_weight_file", default="./DR_weight/Coarse_joint.pkl")

parser.add_argument("--Save_path", default="./DR_result/Coarse_joint/")
parser.add_argument("--Real_Save_path", default="./DR_result/ICSC_LD_LG_Real/")

parser.add_argument("--Drop_musk_weight_file", default="./DR_weight/Rain_drop_musk_5_155.pkl")

parser.add_argument("--GamaG", type=float, default=0.001, help='weight of lossG')
parser.add_argument("--GamaDeep", type=float, default=0.005, help='weight of loss_G')
parser.add_argument("--GamaB", type=float, default=0.001, help='weight of loss_B')
parser.add_argument("--GamaI", type=float, default=0.001, help='weight of loss_I')
parser.add_argument("--GamaS", type=float, default=100, help='weight of loss_Style')

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


def heavy_rainy_name_list(rainy_name_list, clean_name_list, label_name_list, rainy_path, clean_path):
    new_rainy_name_list = []
    new_clean_name_list = []
    new_label_name_list = []
    for i in range(len(rainy_name_list)):
        rainy_image = cv2.imread(rainy_path + rainy_name_list[i].decode())[:, 64:1984, :]
        # print(rainy_image.shape)
        clean_image = cv2.imread(clean_path + clean_name_list[i].decode())[:, 64:1984, :]

        rainy_drop_musk = gen_drop_musk(rainy_image, clean_image, gama=5)

        # print(np.sum(rainy_drop_musk)/(rainy_drop_musk.shape[0]*rainy_drop_musk.shape[1]))

        if (np.sum(rainy_drop_musk) / (rainy_drop_musk.shape[0] * rainy_drop_musk.shape[1]) >= 0.3):
            new_rainy_name_list.append(rainy_name_list[i])
            new_clean_name_list.append(clean_name_list[i])
            new_label_name_list.append(label_name_list[i])
    return new_rainy_name_list, new_clean_name_list, new_label_name_list


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
        # rainy_drop_musk = np.asarray(derain_transform(rainy_drop_musk))

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


############################################################


#############################################################
def expand_label_tensor(label_tesnor):
    B, H, W = label_tesnor.shape
    for i in range(B):
        expanded_label = torch.zeros(1, 20, H, W).cuda()
        for h in range(H):
            for w in range(W):
                expanded_label[0][int(label_tesnor[i][h][w])][h][w] = 1.
        if (i == 0):
            output = expanded_label
        else:
            output = torch.cat((output, expanded_label), 0)
    return output


'''
def iou_loss(SS_output, expanded_label_tesnor):
    B, C, H, W = expanded_label_tesnor.shape
    IoU = 0.0
    for i in range(B):
        for c in range(C):
            Iand1 = torch.sum(SS_output[i, c, :, :] * expanded_label_tesnor[i, c, :, :])
            Ior1 = torch.sum(SS_output[i, c, :, :]) + torch.sum(expanded_label_tesnor[i, c, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            IoU = IoU + (1 - IoU1)
    return IoU / (B * C)
'''


def iou_loss(SS_output, SS_output_GT):
    B, C, H, W = SS_output.shape
    beta = 1000
    for c in range(C):
        if (c == 0):
            index_tensor = torch.ones((B, 1, H, W)).cuda() * c
        else:
            index_tensor = torch.cat((index_tensor, torch.ones((B, 1, H, W)).cuda() * c), 1)
    index_tensor.type(torch.FloatTensor)
    SS_output_GT = SS_output_GT * beta
    SS_output_GT = F.softmax(SS_output_GT, dim=1)
    SS_output_GT = torch.sum(SS_output_GT * index_tensor, 1)
    SS_output_GT = SS_output_GT.trunc().type(torch.LongTensor).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()
    return ce_loss(SS_output, SS_output_GT)


def Boundary_loss(SS_output, SS_output_GT):
    B, C, H, W = SS_output.shape
    Lapulas = LaPulas_Fliter().cuda()
    beta = 1000
    # SS_output_map = Variable(torch.zeros((B,1,H,W)).cuda(), requires_grad=True)
    # SS_output_GT_map = Variable(torch.zeros((B,1,H,W)).cuda(), requires_grad=False)
    for c in range(C):
        if (c == 0):
            index_tensor = torch.ones((B, 1, H, W)).cuda() * (c + 1)
        else:
            index_tensor = torch.cat((index_tensor, torch.ones((B, 1, H, W)).cuda() * (c + 1)), 1)
    index_tensor.type(torch.FloatTensor)
    SS_output = SS_output * beta
    SS_output_GT = SS_output_GT * beta
    SS_output = F.softmax(SS_output, dim=1)
    SS_output_GT = F.softmax(SS_output_GT, dim=1)
    SS_output = torch.sum(SS_output * index_tensor, 1).unsqueeze(1)
    SS_output_GT = torch.sum(SS_output_GT * index_tensor, 1).unsqueeze(1)
    SS_output = Lapulas(SS_output)
    SS_output_GT = Lapulas(SS_output_GT)
    # gt_map = SS_output_GT[0, 0]
    # print(gt_map.shape)
    # print(torch.max(gt_map), torch.min(gt_map))

    # print("##################################")
    # mse_loss = nn.MSELoss().cuda()
    # bce_loss = nn.BCELoss().cuda()
    l1_loss = nn.L1Loss().cuda()

    # cv2.imwrite("./boundary.png", gt_map.cpu().numpy() * 255)
    loss = l1_loss(SS_output / torch.max(SS_output), SS_output_GT / torch.max(SS_output_GT))

    return loss


def Deeplab_loss(low_level_feat, aspp_feat_list, low_level_feat_gt, aspp_feat_list_gt):
    criterion = nn.L1Loss()
    loss = 0
    for i in range(len(aspp_feat_list)):
        loss += criterion(aspp_feat_list[i], aspp_feat_list_gt[i].detach())
    loss = loss / len(aspp_feat_list)
    loss += 1 / 10 * criterion(low_level_feat, low_level_feat_gt.detach())
    return loss


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)


def Style_loss(low_level_feat, aspp_feat_list, low_level_feat_gt, aspp_feat_list_gt):
    criterion = nn.L1Loss()
    loss = 0
    for i in range(len(aspp_feat_list)):
        loss += criterion(gram_matrix(aspp_feat_list[i]), gram_matrix(aspp_feat_list_gt[i].detach()))
    loss = loss / len(aspp_feat_list)
    loss += 1 / 10 * criterion(gram_matrix(low_level_feat), gram_matrix(low_level_feat_gt.detach()))
    return loss


##########################################################
def derain_loss(derain_output_tensor, clean_tensor):
    criterion = SSIM().cuda()
    criterion1 = nn.L1Loss().cuda()
    loss = 0.85 * (1 - criterion(derain_output_tensor,
                                 clean_tensor)) + (1 - 0.85) * criterion1(derain_output_tensor, clean_tensor)
    return loss


def derain_loss_2(Refine_residual, gt_residual):
    criterion1 = nn.MSELoss().cuda()
    loss = criterion1(Refine_residual, gt_residual)
    return loss


##########################################################
def Discriminator_Loss(logits_real, logits_fake):
    N = logits_real.size()
    true_labels = torch.Tensor(torch.zeros(N)).cuda()
    Bce_loss = nn.BCEWithLogitsLoss().cuda()
    Ico_loss = Bce_loss(logits_real, true_labels)
    Ien_loss = Bce_loss(logits_fake, 1 - true_labels)
    loss = Ico_loss + Ien_loss
    return loss


def Generator_loss(logits_fake):
    N = logits_fake.size()
    true_labels = torch.Tensor(torch.zeros(N)).cuda()
    Bce_loss = nn.BCEWithLogitsLoss().cuda()
    loss = Bce_loss(logits_fake, true_labels)
    return loss


'''
def Discriminator_Loss(logist_real, logist_fake):
    criterionGAN = GANLoss(real_label=1.0, fake_label=0.0)
    loss_fake = criterionGAN(logist_fake, is_real=False)
    loss_real = criterionGAN(logist_real,is_real=True)
    return loss_fake + loss_real

def Generator_loss(logist_fake):
    criterionGAN = GANLoss(real_label=1.0, fake_label=0.0)
    loss_fake = criterionGAN(logist_fake, is_real=False)
    return -1*loss_fake
'''


##########################################################
##########################################################
##########################################################
def train(args, Coarse_Net_model, Rain_drop_model, SS_model):
    Train_clean_image_name = read_data(args.Data_path + "Train_Clean_image_name.h5")
    Train_rain_image_name = read_data(args.Data_path + "Train_Rainy_image_name.h5")
    Train_label_image_name = read_data(args.Data_path + "Train_Label_image_name.h5")

    rainy_name_list = Train_rain_image_name
    clean_name_list = Train_clean_image_name
    label_name_list = Train_label_image_name

    optimizer_D = torch.optim.Adam(Coarse_Net_model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    #scheduler1 = MultiStepLR(optimizer_D, milestones=args.milestone, gamma=0.5)
    # scheduler2 = MultiStepLR(optimizer_A, milestones=args.milestone, gamma=0.2)

    counter = 0

    start_time = time.time()
    
    #weight_name = "./DR_weight/Coarse_Nets/Coarse_Nets_183.pkl"
    Coarse_Net_model.load_state_dict(torch.load(args.DR_weight_file))
    
    for ep in range(args.epoch):
        #scheduler1.step(ep)
        # scheduler2.step(ep)
        batch_idxs = len(rainy_name_list) // args.batch_size
        for idx in range(0, batch_idxs):
            # batching......
            batch_rainy_name_list = rainy_name_list[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_clean_name_list = clean_name_list[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_label_name_list = label_name_list[idx * args.batch_size: (idx + 1) * args.batch_size]
            Train_rainy_tensor, Train_clean_tensor, Train_label, Train_rain_drop_musk = batch_tensor(
                batch_rainy_name_list,
                batch_clean_name_list,
                batch_label_name_list,
                args.rain_data_path,
                args.clean_data_path, args.label_path,
                args.image_size)

            counter += 1

            optimizer_D.zero_grad()
            out_musk = Rain_drop_model(Train_rainy_tensor)
            
            Train_input = torch.cat((Train_rainy_tensor, out_musk), dim=1)
            coarse_output = Coarse_Net_model(Train_input)
            derain_output = coarse_output
            SS_output, low_level_feat, aspp_feat_list = SS_model(Normalize_to_SS(derain_output))
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            Nomrmalized_Train_clean_tensor = (Train_clean_tensor - mean) / std
            SS_output_GT, low_level_feat_gt, aspp_feat_list_gt = SS_model(Nomrmalized_Train_clean_tensor)

            Loss_Deep = Deeplab_loss(low_level_feat, aspp_feat_list, low_level_feat_gt, aspp_feat_list_gt)

            Loss_D = derain_loss(coarse_output, Train_clean_tensor)

            loss = Loss_D + args.GamaDeep * Loss_Deep
            loss.backward()
            optimizer_D.step()

            loss2 = 0
            ######################################

            if counter % 2 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss1: [%.8f], loss2: [%.8f]" \
                      % ((ep + 1), counter, time.time() - start_time, loss, loss2))
                print("Loos_D: ", Loss_D)
                print("Loss_Deep: ", Loss_Deep)

            if counter % 50 == 0:
                torch.save(Coarse_Net_model.state_dict(), args.DR_weight_file)
        torch.save(Coarse_Net_model.state_dict(), "./DR_weight/Coarse_joint/Coarse_Nets_" + str(ep) + ".pkl")


############################################################################################
############################################################################################
############################################################################################
############################################################################################
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


def test(args, derain_model):
    # Test_clean_image_name = read_data(args.Data_path + "Test_Clean_image_name.h5")
    Test_rain_image_name = read_data(args.Data_path + "Test_Rainy_image_name.h5")
    # Test_label_image_name = read_data(args.Data_path + "Test_Label_image_name.h5")

    # input_name_list = Test_clean_image_name
    input_name_list = Test_rain_image_name
    # label_name_list = Test_label_image_name

    # input_path = args.clean_data_path
    input_path = args.rain_data_path
    # label_path = args.label_path

    derain_model.load_state_dict(torch.load(args.DR_weight_file))
    derain_model.eval()

    for index in range(len(input_name_list)):
        input_image = cv2.imread(input_path + input_name_list[index].decode())[:, 64:1984, :]
        input_tensor_list = cut_batch_tensor(input_image, 2)
        output_list = []
        for l in range(len(input_tensor_list)):
            output_tensor_list = derain_model(input_tensor_list[l])
            output_tensor_list = [output_tensor_list]
            output_tensor_list = output_list_to_cpu(output_tensor_list)
            output_list.append(output_tensor_list)
        output_image_list = merge_tensor_to_image(output_list)

        for i in range(len(output_image_list)):
            save_path = args.Save_path + "DR_predict/" + str(i + 1) + "/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv2.imwrite(save_path + str(index) + "_DR.png", output_image_list[i])
        cv2.imwrite(args.Save_path + "input/" + str(index) + "_input.png", input_image)
        # print("DR_", str(index))
        torch.cuda.empty_cache()


###########################################################################################
###########################################################################################
###########################################################################################
def test_real(args, derain_model):
    derain_model.load_state_dict(torch.load(args.DR_weight_file))
    derain_model.eval()
    data_path = "/home/User1/gmx/derain/Semantic_Segmentation_benchmark/pytorch-deeplab-xception-master/Real_data/images/"
    image_file_list = os.listdir(data_path)
    for index in range(len(image_file_list)):
        if (index % 10 == 0):
            input_image = cv2.imread(data_path + image_file_list[index])
            input_image = cv2.resize(input_image, (1920, 1024))
            print(input_image.shape)
            input_tensor_list = cut_batch_tensor(input_image, 2)
            output_list = []
            for l in range(len(input_tensor_list)):
                output_tensor_list = derain_model(input_tensor_list[l])
                output_tensor_list = [output_tensor_list]
                output_tensor_list = output_list_to_cpu(output_tensor_list)
                output_list.append(output_tensor_list)
            output_image_list = merge_tensor_to_image(output_list)

            for i in range(len(output_image_list)):
                save_path = args.Real_Save_path + "DR_predict/" + str(i + 1) + "/"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                cv2.imwrite(save_path + str(index) + "_DR.png", output_image_list[i])
            cv2.imwrite(args.Real_Save_path + "input/" + str(index) + "_input.png", input_image)
            print("DR_", str(index))
            torch.cuda.empty_cache()


###########################################################################################
###########################################################################################
###########################################################################################
def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def SS_cut_batch_tensor(img):
    (H, W, C) = img.shape
    sub_img1 = img[:, 0:W // 2, :]
    sub_img2 = img[:, W // 2:, :]
    sub_img1 = derain_transform(sub_img1).unsqueeze(0).type(torch.FloatTensor).cuda()
    sub_img2 = derain_transform(sub_img2).unsqueeze(0).type(torch.FloatTensor).cuda()
    batch_tensor = torch.cat((sub_img1, sub_img2))
    norm_batch_tensor = (batch_tensor - torch.min(batch_tensor)) / (
            torch.max(batch_tensor) - torch.min(batch_tensor))
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    norm_batch_tensor = (norm_batch_tensor - mean) / std
    return norm_batch_tensor


def SS_merge_tensor_to_image(output_tensor):
    for i in range(output_tensor.shape[0]):
        img = output_tensor[i].cpu().detach().numpy()
        if (i == 0):
            merge_img = img
        else:
            # print(merge_img.shape)
            merge_img = np.concatenate((merge_img, img), 2)
    return merge_img


def test_multi_weight(args, Coarse_Net_model, Rain_drop_model, SS_model):
    Test_clean_image_name = read_data(args.Data_path + "Test_Clean_image_name.h5")
    Test_rain_image_name = read_data(args.Data_path + "Test_Rainy_image_name.h5")
    Test_label_image_name = read_data(args.Data_path + "Test_Label_image_name.h5")

    clean_input_name_list = Test_clean_image_name
    rainy_input_name_list = Test_rain_image_name
    label_name_list = Test_label_image_name

    clean_input_path = args.clean_data_path
    rainy_input_path = args.rain_data_path
    label_path = args.label_path

    weight_path = "./DR_weight/Coarse_joint/"
    weight_name_list = os.listdir(weight_path)

    number_list = [i for i in range(183, 199)]
    # for k in range(len(weight_name_list)):
    for k in number_list:
        print("epoch :", k)
        #weight_name = weight_path + "Coarse_Nets_" + str(k) + ".pkl"
        weight_name = "./DR_weight/ICSC_LD_LDeep/ICSC_LD_LDeep_" + str(k) + ".pkl"
        Coarse_Net_model.load_state_dict(torch.load(weight_name))
        Coarse_Net_model.eval()
        argv_psnr = 0
        argv_ssim = 0
        TP = [0 for i in range(19)]
        FN = [0 for i in range(19)]
        FP = [0 for i in range(19)]
        for index in range(len(rainy_input_name_list)):
            print(index)
            # derain_testing...........
            # print(rainy_input_name_list[index].decode(), clean_input_name_list[index].decode(),label_name_list[index].decode())
            input_image = cv2.imread(rainy_input_path + rainy_input_name_list[index].decode())[:, 64:1984, :]
            input_tensor_list = cut_batch_tensor(input_image, 2)
            output_list = []
            for l in range(len(input_tensor_list)):
                input_tensor = input_tensor_list[l]
                #output_musk = Rain_drop_model(input_tensor)
                #derain_input = torch.cat((input_tensor, output_musk), 1)
                derain_input = input_tensor
                coarse_output = Coarse_Net_model(derain_input)
                output_tensor = coarse_output
                output_tensor_list = [output_tensor]
                output_tensor_list = output_list_to_cpu(output_tensor_list)
                output_list.append(output_tensor_list)
            output_image_list = merge_tensor_to_image(output_list)
            derain_image = output_image_list[0]
            clean_image = cv2.imread(clean_input_path + clean_input_name_list[index].decode())[:, 64:1984, :]
            
            # SS_testing..........
            label = cv2.imread(label_path + label_name_list[index].decode(), 0)[:, 64:1984]
            label[label == 255] = 19
            H, W, C = derain_image.shape
            input_tensor = SS_cut_batch_tensor(derain_image)
            SS_output, low_level_feat, aspp_feat = SS_model(input_tensor)
            output_image = SS_merge_tensor_to_image(SS_output)
            # print(output_image.shape)
            predict = np.zeros([H, W])
            for i in range(H):
                for j in range(W):
                    predict_class = np.argmax(output_image[:, i, j])
                    predict[i, j] = predict_class
                    if (int(predict[i, j]) == int(label[i, j]) and int(label[i, j]) != 19):
                        TP[int(label[i, j])] += 1
            FN = [FN[i] + sum(sum(label == i)) for i in range(19)]
            FP = [FP[i] + sum(sum(predict == i)) for i in range(19)]
            # print("FN",FN)
            # print("FP",FP)
            # print("TP",TP)
            
            cv2.imwrite("./buffer1.png", derain_image)
            derain_image = cv2.imread("./buffer1.png")
            psnr = calc_psnr(derain_image, clean_image)
            ssim = calc_ssim(derain_image, clean_image)

            argv_psnr += psnr
            argv_ssim += ssim
            
            cv2.imwrite("./DR_result/Coarse_joint/derain/" + str(index) + "_derain.png", derain_image)
            cv2.imwrite("./DR_result/Coarse_joint/SS/"+ str(index) + "_SS.png", predict)
            
            if (index == 7):
                cv2.imwrite(args.Save_path + "Multi_weight/epoch_" + str(k) + ".png", derain_image)
            torch.cuda.empty_cache()
        
        miou = 0
        non_class = 0
        for Class in range(19):
            if (FN[Class] == 0 and FP[Class] == 0):
                non_class += 1
                continue
            else:
                miou += TP[Class] / (FN[Class] + FP[Class] - TP[Class])
        miou = miou / (19 - non_class)
        print("mIou :", miou)
        
        print("argv_psnr:", argv_psnr / len(rainy_input_name_list))
        print("argv_ssim:", argv_ssim / len(rainy_input_name_list))


args = parser.parse_args()

Coarse_Net_model = create_coarse_nets()
Coarse_Net_model = nn.DataParallel(Coarse_Net_model)

Rain_drop_musk_net = rain_drop_musk_net(n_blocks=5).cuda()
Rain_drop_model = nn.DataParallel(Rain_drop_musk_net)
Rain_drop_model.load_state_dict(torch.load(args.Drop_musk_weight_file))
Rain_drop_model.eval()

SS_model = DeepLab(backbone=args.backbone, output_stride=args.output_stride).cuda()
SS_model = nn.DataParallel(SS_model)
SS_model.load_state_dict(torch.load(args.SS_weight_file))
SS_model.eval()

if (args.is_Train):
    train(args, Coarse_Net_model, Rain_drop_model, SS_model)
    test_multi_weight(args, Coarse_Net_model, Rain_drop_model, SS_model)
else:
    if (args.Test_Real):
        test_real(args, derain_model)
    else:
        if (args.Test_multi_weight):
            test_multi_weight(args, Coarse_Net_model, Rain_drop_model, SS_model)
        else:
            test(args, derain_model)
