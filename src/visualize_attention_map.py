import sys


import os

import PIL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import cv2
from PIL import Image
from torchvision import transforms

from ast_based_novelty_visualization import adast

from spectrogram_plots import convert_to_spectrogram_and_save
from sklearn import preprocessing
import gc


transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_attention_map(input_location,img, model,get_mask=False):

    att_mat = model.get_ast_embedding_single_file(input_location)



    att_mat = torch.stack(att_mat).squeeze(1)

    print(att_mat.detach().numpy().shape)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)


    print(att_mat.detach().numpy().shape)


    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)



    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    mask = (v[0, 2:].reshape(101, 12).detach().numpy()+v[0, 2:].reshape(101, 12).detach().numpy())/2


    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        mask_avg = np.average(mask)
        mask=mask/mask_avg
        result = (mask * img).astype("uint8")
        for row in result:
            for culumn in row:
                for i in range(3):
                    if culumn[i]>=255:
                        culumn[i]=255

                culumn[3]=255

    return result


def plot_attention_map(original_img, att_map):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    plt.show()




base_directory="../../dev_data/"
output_base_directory="../results/spectrograms/"
for machine in os.listdir(base_directory):
    for domain in os.listdir(base_directory+"/"+machine):
        input_directory = base_directory + machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        for filename in os.listdir(input_directory):
            if filename.endswith(".wav"):
                file_location = os.path.join(input_directory, filename)
                sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + ".png"
                convert_to_spectrogram_and_save(file_location, output_location)
                gc.collect()

        print(machine+" "+domain+" done")


#adast_mdl = adast()

#file_location=os.path.join("../../dev_data/fan/train/section_00_source_train_normal_0025_strength_1_ambient.wav")
#output_location=os.path.join("../results/spectrograms/fan/train/section_00_source_train_normal_0025_strength_1_ambient.png")
#convert_to_spectrogram_and_save(file_location,output_location)

#img=Image.open(output_location)


#result = get_attention_map(file_location,img,adast_mdl)

#plot_attention_map(img, result)

#plot_attention_map(img1, result1)