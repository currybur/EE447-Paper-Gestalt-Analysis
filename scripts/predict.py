# -*- coding:utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import lightgbm as lgb
import PIL.Image as Image
from torchvision import models
from torchvision import transforms
from torch.nn import functional as F
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams

sys.path.append('..')
from nn_process import save_jpg, save_pages, paper_to_image
from lgb_process import get_pdf_meta

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
NET_WEIGHT = 0.5
TEST_DIR = '../dataset/test/'
TEST_DIR1 = '../dataset_new/test/'

def get_nn_model(mode='overall'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = models.vgg19_bn(pretrained=False)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 2)

    # model = models.densenet121(pretrained=False)
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, 2)
    # model = models.resnet18(pretrained=False)
    # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    if mode == 'overall':
        ckp_path = OUTPUT_DIR + 'nn_output/model_best_vgg19bn_ov.pth.tar'
    elif mode == 'page':
        model = models.resnet18(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        ckp_path = OUTPUT_DIR + 'nn_output/model_best_res18_page.pth.tar'
    checkpoint = torch.load(ckp_path)
    d = checkpoint['state_dict']
    d = {k.replace('module.', ''): v for k, v in d.items()}
    model.load_state_dict(d)

    # gestalt_model = models.resnet18(pretrained=False)
    # gestalt_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # gestalt_model.fc = nn.Linear(gestalt_model.fc.in_features, 2)
    # gestalt_model.load_state_dict(torch.load(OUTPUT_DIR + 'nn_output/PaperNet.pth'))
    # gestalt_model = gestalt_model.to(device)
    # model = gestalt_model
    #
    model = model.to(device)

    return model


def get_lgb_model():
    return lgb.Booster(model_file=OUTPUT_DIR + 'lgb_output/lgb_model.txt')


def load_img(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = trans(img)
    img = img.to(device)
    img = torch.unsqueeze(img, 0)
    return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="pdf_path")
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    args = parser.parse_args()
    return args


def draw_CAM(pdf_path, model, save_path="./CAM_pages.jpg", visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt = tmp_dir + pdf_path.split('/')[-1].replace('.pdf','')
    page_num = save_pages(pdf_path, tgt)

    fromImage = Image.open(tgt + '0.png')
    width, height = fromImage.size
    toImage = Image.new('RGB', (width * page_num, height * 1))
    print("---------totally {} pages----------".format(page_num))
    for i in range(page_num):
        print("processing page {} .....".format(i+1))
        tgt_path = tgt + '%s.png'%i
        img_path = tgt_path
        img = load_img(tgt_path)
        # logit = model(img)

        # img = Image.open(img_path).convert('RGB')
        # if transform:
        #     img = transform(img)
        # img = img.unsqueeze(0)

        # 获取模型输出的feature/score
        model.eval()
        # features = model.features(img)
        # pooled = model.avgpool(features).view(1,25088)
        # output = model.classifier(pooled)

        # for resnet18
        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )(img)
        pooled = model.avgpool(features).view(1, 512)
        output = model.fc(pooled)

        # 为了能读取到中间梯度定义的辅助函数
        def extract(g):
            global features_grad
            features_grad = g

        # 预测得分最高的那一类对应的输出score
        pred = torch.argmax(output).item()
        pred_class = output[:, pred]

        features.register_hook(extract)
        pred_class.backward()  # 计算梯度

        grads = features_grad  # 获取梯度

        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

        # 此处batch size默认为1，所以去掉了第0维（batch size维）
        pooled_grads = pooled_grads[0]
        features = features[0]
        # 512是最后一层feature的通道数
        for j in range(512):
            features[j, ...] *= pooled_grads[j, ...]

        # 以下部分同Keras版实现
        heatmap = features.detach().to(torch.device('cpu')).numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # 可视化原始热力图
        if visual_heatmap:
            plt.matshow(heatmap)
            plt.show()

        img = cv2.imread(img_path)  # 用cv2加载原始图像
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
        # cv2.imwrite(save_path, heatmap)
        fromImage = Image.open(save_path)
        toImage.paste(fromImage, (i * width, 0))
        fromImage.close()
        os.remove(save_path)

    for i in range(page_num):
        fname = tgt + '%s.png'%i
        if os.path.exists(fname):
            os.remove(fname)

    toImage.save(save_path)
    return save_path


def draw_CAM_overall(pdf_path, model, save_path="./CAM_overall.jpg", transform=None, visual_heatmap=False):
    '''
      绘制 Class Activation Map
      :param model: 加载好权重的Pytorch model
      :param img_path: 测试图片路径
      :param save_path: CAM结果保存路径
      :param transform: 输入图像预处理方法
      :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
      :return:
    '''
    # 讲pdf存为img
    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt_path = tmp_dir + 'image.jpg'
    # print(tgt_path)
    save_jpg(pdf_path, tgt_path, tmp_dir)  # complete paper
    img_path = tgt_path

    # 图像加载&预处理
    # img = Image.open(img_path).convert('RGB')
    # if transform:
    #     img = transform(img)
    # img = img.unsqueeze(0)
    img = load_img(img_path)

    # 获取模型输出的feature/score
    model.eval()

    features = model.features(img)
    pooled = model.avgpool(features).view(1, 25088)
    output = model.classifier(pooled)

    # features = nn.Sequential(
    #     model.conv1,
    #     model.bn1,
    #     model.relu,
    #     model.maxpool,
    #     model.layer1,
    #     model.layer2,
    #     model.layer3,
    #     model.layer4,
    # )(img)
    # pooled = model.avgpool(features).view(1, 512)
    # output = model.fc(pooled)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    # print(output.size())
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for j in range(512):
        features[j, ...] *= pooled_grads[j, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().to(torch.device('cpu')).numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    # cv2.imwrite(save_path, heatmap)

    if os.path.exists(tgt_path):
        os.remove(tgt_path)

    return save_path


def predict_overall(pdf_path, model, lgb_model):
    # lgb predict
    figures, tables, formulas, cnt = get_pdf_meta(pdf_path)
    meta = figures + tables + formulas + [cnt]
    gbm_score = lgb_model.predict(np.array([meta]))[0]

    # nn predict
    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt_path = tmp_dir + 'image.jpg'
    # print(tgt_path)
    save_jpg(pdf_path, tgt_path, tmp_dir)
    # tgt_path = paper_to_image(pdf_path)
    img = load_img(tgt_path)
    logit = model(img)
    probs = F.softmax(logit.to(torch.device("cpu"))/5, dim=1).data.squeeze()
    nn_score = probs[0].numpy()

    score = NET_WEIGHT * nn_score + (1 - NET_WEIGHT) * gbm_score
    # score = nn_score
    # score = probs
    return round(score*100, 0)
    # return score


def predict_page(img_path, model):
    img = load_img(img_path)
    logit = model(img)
    probs = F.softmax(logit.to(torch.device("cpu")), dim=1).data.squeeze()
    nn_score = probs[1].numpy()

    return round(nn_score*100, 1)


def predict_pages(pdf_path, model):  # 用于对pdf的每页打分各自
    # lgb predict
    # figures, tables, formulas, cnt = get_pdf_meta(pdf_path)
    # meta = figures + tables + formulas + [cnt]
    # gbm_score = lgb_model.predict(np.array([meta]))[0]

    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt = tmp_dir + pdf_path.split('/')[-1].replace('.pdf','')
    page_num = save_pages(pdf_path, tgt)
    scores = []
    for i in range(page_num):
        img_path = tgt + '%s.png' % i
        img = load_img(img_path)
        logit = model(img)
        probs = F.softmax(logit, dim=1).data.squeeze()
        nn_score = probs[1].numpy()
        scores.append(nn_score)

    nn_score = sum(scores)/page_num
    # score = NET_WEIGHT * nn_score + (1 - NET_WEIGHT) * gbm_score
    score = nn_score
    return round(score*100, 1)


def show_score_dist(model):
    score_dict = {}

    for conf in ['conference', 'workshop']:
        score_dict[conf] = []
        path = TEST_DIR + conf + '/'
        lists = os.listdir(path)
        l = len(lists)
        print('total {} files in {}'.format(l, conf))
        ct = 0
        for pt in lists:
            try:
                ct += 1
                print("processing {}, {}/{}".format(pt, ct, l))
                img_path = path + pt
                score = predict_page(img_path, model)  # 直接使用处理过的图片
                # score = predict_pages(img_path, model)
                score_dict[conf].append(score)
            except:
                print("------fail {}".format(pt))
                pass

    params = dict(histtype='stepfilled', alpha=0.3,bins=40)

    for conf, scores in score_dict.items():
        plt.hist(scores, **params)
    plt.xlabel('score')
    plt.ylabel('number')
    plt.savefig('./plot.png')


def count_numbers():  # figure, table, formula, page_num
    score_dict = {}

    for conf in ['conference', 'workshop']:
        score_dict[conf] = {'figure':[],'table':[],'formula':[],'page_num':[]}
        path = TEST_DIR + conf + '/'
        lists = os.listdir(path)
        l = len(lists)
        print('total {} files in {}'.format(l, conf))
        ct = 0
        for pt in lists:
            try:
                # if ct > 20:
                #     break
                ct += 1
                print("processing {}, {}/{}".format(pt, ct, l))
                img_path = path + pt
                # score = predict_page(img_path, model)
                # score = predict_pages(img_path, model)
                fg, tb, fm, pn = get_pdf_meta(img_path)
                score_dict[conf]['figure'].append(np.var(fg))
                score_dict[conf]['table'].append(np.var(tb))
                score_dict[conf]['formula'].append(np.var(fm))
                score_dict[conf]['page_num'].append(pn)
                # print(score_dict[conf])
            except:
                print("{} failed".format(pt))
                pass


    for sd in ['figure','table','formula','page_num']:
        for conf, scores in score_dict.items():
            # print(scores[sd])
            params = dict(histtype='bar', alpha=0.5,bins=min(20,len(set(scores[sd]))),align='mid')
            plt.hist(scores[sd], **params)
        plt.xlabel("{} number".format(sd))
        plt.ylabel('number of papers')
        plt.savefig('./plot_{}.png'.format(sd))
        plt.close()


def main():
    # pdf_path = "../web/uploads/upload.pdf"
    # pdf_path = "upload.pdf"
    pdf_path = "../dataset_new/test/conference/Chuang_Learning_to_Act_CVPR_2018_paper.pdf"
    lgb_model = get_lgb_model()

    nn_model_overall = get_nn_model('overall')
    nn_model_page = get_nn_model('page')

    score = predict_overall(pdf_path, nn_model_overall, lgb_model)  # lgb+nn的分数
    CAM_page = draw_CAM(pdf_path, nn_model_page)  # 单页合成热力图路径
    CAM_ov = draw_CAM_overall(pdf_path, nn_model_overall)  # 整体热力图路径

    return score, CAM_ov


def threshold_test(model, lgb_model):
    thress = [i for i in range(40, 60, 1)]
    confuse_matrixs = [[[0, 0] for _ in range(2)] for __ in range(len(thress))]
    for conf, label in [['conference',1], ['workshop',0]]:
        path = TEST_DIR1 + conf + '/'
        lists = os.listdir(path)
        l = len(lists)
        print('total {} files in {}'.format(l, conf))
        ct = 0
        for pt in lists:
            try:
                ct += 1
                # print("processing {}, {}/{}".format(pt, ct, l))
                # img_path = path + pt
                # score = predict_page(img_path, model)  # 直接使用处理过的图片
                pdf_path = path + pt
                score = predict_overall(pdf_path, model, lgb_model)
                for i in range(len(thress)):
                    if score > thress[i]:
                        confuse_matrixs[i][label][1] += 1
                    else:
                        confuse_matrixs[i][label][0] += 1
            except:
                print("------fail {}".format(pt))
                pass

    for i in range(len(confuse_matrixs)):
        matx = np.array(confuse_matrixs[i])
        print("-------------threshold {} has confuse matrix \n{}, accuracy {}".format(thress[i], matx, (matx[0][0]+matx[1][1])/matx.sum()))

    # return confuse_matrixs


if __name__ == '__main__':
    args = parse_args()
    try:
        pdf_path = args.pdf_path
    except:
        print("no pdf chosen")

    print(main())

    # nn_model = get_nn_model('overall')
    # lgb_model = get_lgb_model()
    # threshold_test(nn_model, lgb_model)
    # show_score_dist(nn_model)
    # draw_CAM(nn_model)
    # count_numbers()

    # score = predict_overall(pdf_path, nn_model, lgb_model)
    # score = predict_pages(pdf_path, nn_model)
    # print('The score of {} is {}'.format(pdf_path, score))


