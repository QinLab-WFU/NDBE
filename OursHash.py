# VTS (CSQ with ViT Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# CSQ basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools import *
from network import *
# from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
#from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
import argparse
import os
import random
import torch
import torch.optim as optim

import time

from network import ResNet,ResNet50

from loss.SoftTriple import SoftTriple

torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        #"dataset": "cifar10",
        #"dataset": "mirflickr",
        #"dataset": "cifar10-2",
        "dataset": "coco",
        #"dataset": "nuswide_21",
        #"dataset": "imagenet",
        # "net": AlexNet,
        # "net_print": "AlexNet",
        # "net":ResNet,
        # "net_print": "ResNet",
        # "net": VisionTransformer,
        # "net_print": "ViT-B_32",
        # "model_type": "ViT-B_32",
        # "pretrained_dir": "pretrainedVIT/ViT-B_32.npz",
        #"net": VisionTransformer,
        #"net": Network,
        "net": ResNet50,
        "net_print": "",
        #"net_print": "ViT-B_32",
        # "model_type": "Con-B_16",
        # "model_type": "ViT-B_32",
        # "pretrained_dir": ViT-B_32.npz",
        # "pretrained_dir": "pretrainedVIT/Conformer_base_patch16.pth",
        "bit_list": [32],
        "optimizer": {"type": optim.Adam,
        "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "device": torch.device("cuda"),
        "save_path": "Checkpoints_Results",
        "epoch":100,
        "test_map": 3,
        "batch_size": 32,
        "resize_size": 256,
        "crop_size": 224,
        "info": "",
        "lambda": 0.1,
        'la':3.0,
        'gamma':0.2,#0.1
        'tau':0.5,
        'margin':0.01,
        'K':2,

    }
    config = config_dataset(config)
    return config
def adjust_learning_rate(optimizer, epoch):
    # decayed lr by 10 every 20 epochs
    if (epoch+1)%20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.01
def train_val(config, bit):
    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    num_classes = config["n_class"]
    hash_bit = bit

    net = ResNet50(bit,pretrained=True)#rensnet50

    net = net.to(device)



    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"],
                             config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(
                                 bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"],
                                config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(
                                    bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"],
                                config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(
                                    bit) + ".txt")
    f = open(results_path, 'a')
    config['dim']=hash_bit
    config['C']=config["n_class"]
    criterion = SoftTriple(dim=hash_bit, cN=num_classes,
                           K=config['K'],
                           la=config['la'],
                           gamma=config['gamma'],
                          tau=config['tau'],
                           margin=config['margin']).to(device)


    total_time=0


    optimizer = optim.Adam([{"params":net.parameters(),"lr":0.0005},
                           {"params":criterion.parameters(),"lr":0.005}
                            ],
                           eps=0.01,weight_decay=0.005)

    for epoch in range(start_epoch, config["epoch"] + 1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        adjust_learning_rate(optimizer, epoch)
        #L_net.set_alpha(epoch)
        train_loss = 0
        start_time=time.time()
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            u=net(image)


            loss1 = criterion(u,label.float())

            loss = loss1#log_loss_it#+i_noise_loss
            optimizer.zero_grad()
            train_loss += loss.item()#+q_loss.item()
            loss.backward()
            optimizer.step()
        total_time+=time.time()-start_time
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f|Traintime:%.6f" % (train_loss,total_time))
        f.write('Train | Epoch: %d | Loss: %.3f | Traintime:%.6f\n' % (epoch, train_loss,total_time))

        if (epoch) % config["test_map"] ==0:
            # print("calculating test binary code......")
            start_encode_time=time.time()

            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            # # print("calculating dataset binary code.......")\
            # trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            encode_time=time.time()-start_encode_time




            print('encodetime:',encode_time)
            f.write('encodetime:%.3f\n' % (encode_time))
            if mAP > Best_mAP:
                Best_mAP = mAP


                f.write('\n')

            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
            f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
                    % (epoch, mAP, Best_mAP))
        '''state = {
            'net': net.state_dict(),
            'Best_mAP': Best_mAP,
            'epoch': epoch,
        }
        torch.save(state, trained_path)'''
    f.close()


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)