import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt

operation_mat = scipy.io.loadmat('./operation.mat')   #骨架提取算子
o1 = operation_mat['o1']
o2 = operation_mat['o2']


def boundary_2d(img,d=0.4,threshold=0.3):
    '''
    Function:
        求距离目标距离为d的边界
    Param:
        img: 输入图像，不含通道数，灰度值范围0-1
        d: 边界值取d,根据图像大小取值
        threshold: 灰度值小于threshold认为是0,建议设为0.2,0.3
    Return:
        img_b:边界
    '''
    img = img.detach().numpy()

    img[img>=threshold]=1
    img[img<threshold]=0
    img_update=np.array(img)
    img_new=np.array(img)
    img_b=np.zeros_like(img)
    [h,w]=img.shape
    #print(h,w)

    for _ in range(d):
        for i in range(1,h-1):  #图像边缘像素不做操作
            for j in range(1,w-1):
                if(img[i,j]==1):
                    value=img[i-1][j-1] + img[i-1][j]*2 + img[i-1][j+1]*4 + img[i][j-1]*8 + img[i][j+1]*16 + img[i+1][j-1]*32 + img[i+1][j]*64 + img[i+1][j+1]*128
                    if(o1[0][int(value)]==1):   # 说明是靠近边界的像素
                        img_new[i,j]=-1
                        img_update[i][j]=0  #因为要操作1、操作2都做完才能删，因此多设了一个矩阵img_update来存放操作1要删的元素
        for i in range(1,h-1):  #图像边缘像素不做操作
            for j in range(1,w-1):
                if(img[i][j]==1):
                    value=img[i-1][j-1] + img[i-1][j]*2 + img[i-1][j+1]*4 + img[i][j-1]*8 + img[i][j+1]*16 + img[i+1][j-1]*32 + img[i+1][j]*64 + img[i+1][j+1]*128
                    if(o2[0][int(value)]==1):   # 说明是靠近边界的像素
                        img_new[i][j]=-1
                        img_update[i][j]=0
        img=np.array(img_update)
    for i in range(1,h-1):  #图像边缘像素不做操作
        for j in range(1,w-1):
            if(img_new[i][j]==-1):
                img_b[i][j]=1

    return(img_b)


def Batch_Boundary_Ious(predicts, labels, threshold, d, b_threshold):
    '''
    Param:
        predicts: [b,c,h,w]
        labels: [b,c,h,w]
        threshold: 小于阈值的灰度会被置于0
        d: boundary宽度
        b_threshold: 灰度值小于threshold认为是0
    Return:
        b_ious: [b,c]
    '''
    b_ious = torch.zeros(predicts.shape[0], predicts.shape[1])  # 为该batch中每张图像存储各类的iou

    zero = torch.zeros_like(predicts)
    predicts = torch.where(predicts < threshold, zero, predicts)

    for i in range(predicts.shape[0]):  # 遍历图像
        for j in range(predicts.shape[1]):  # 遍历类别，包括背景
            if (labels[i][j].sum() == 0):  # 说明没有这个类别
                b_iou = -1  # 用于标记这个类别不存在iou，方便计算非空标签数量
            else:
                pre_boundary = boundary_2d(predicts[i][j], d, b_threshold)
                label_boundary = boundary_2d(labels[i][j], d, b_threshold)
                pre_boundary = torch.tensor(pre_boundary)
                label_boundary = torch.tensor(label_boundary)
                intersection = torch.sum(torch.logical_and(pre_boundary, label_boundary))
                union = torch.sum(torch.logical_or(pre_boundary, label_boundary))
                b_iou = intersection / union
            b_ious[i][j] = b_iou
    return b_ious


def Boundary_IoU(all_bious):
    valid_num = torch.zeros(1, all_bious.shape[1])  # 存储每种类别的有效标签的个数
    for i in range(all_bious.shape[1]):
        valid_num[0][i] = torch.tensor(np.sum(all_bious[:, i].numpy() >= 0))
        # valid_num=np.sum(all_bious.numpy()>=0)#计算非空标签的数量

    zero = torch.zeros_like(all_bious)
    all_bious_ = torch.where(all_bious < 0, zero, all_bious)  # all_bious中小于0的（即-1，不存在标签的情况）用zero(0)替换,否则不变

    sum_type_biou = np.divide(torch.sum(all_bious_, dim=0), valid_num)  # 点除，相对于all_ious_的每列和除以valid_num的相应列的值

    biou = torch.sum(all_bious_) / torch.sum(valid_num)
    print("各类的Boundary IoU：")
    print(sum_type_biou)
    print("总Boundary IoU：")
    print(biou)


Threshold=0.5
boundary_threshold=0.5
boundary_d=3


def evaluation(y_p, y, x):
    with torch.no_grad():
        all_boundary_ious = Batch_Boundary_Ious(y_p, y, Threshold, boundary_d, boundary_threshold)
        all_boundary_ious = all_boundary_ious[torch.arange(all_boundary_ious.size(0)) != 0]
        Boundary_IoU(all_boundary_ious)
        y = y.to(torch.device('cpu'))
        mIOU = []
        y = y.squeeze()
        y_p = y_p.squeeze()

        for i in range(y.shape[0]):
            intersection = np.logical_and(y[i], y_p[i])
            union = np.logical_or(y[i], y_p[i])
            iou = intersection.sum() * 1.0 / union.sum()
            mIOU.append(iou)
        print(np.mean(mIOU))


