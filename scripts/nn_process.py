# -*- coding:utf-8 -*-

import os
import fitz
import random
import shutil
from tqdm import tqdm
import PIL.Image as Image
from pdf2image import convert_from_path
import cv2

TGT_INPUT_DIR = '../input/'
INPUT_DIR = '../dataset_new/'
OUTPUT_DIR = '../output/'
TEST_SIZE = 0.2

# convert the first W*H pages of pdf to one image,
# which has H rows and W pages in each row
W = 4
H = 2


def save_jpg(src, tgt, tmp_dir):
    doc = fitz.open(src)
    # if doc.pageCount < W * H + 1: # pagenum>=W*H without considering reference page
    #     return

    for pg in range(min(doc.pageCount, W * H)):
        page = doc[pg]
        zoom = 40
        rotate = 0
        trans = fitz.Matrix(zoom / 100.0, zoom / 100.0).preRotate(rotate)

        # create raster image of page (non-transparent)
        pm = page.getPixmap(matrix=trans, alpha=False)

        # write a PNG image of the page
        pm.writePNG(tmp_dir + '%s.png' % pg)

    index = 0
    fromImage = Image.open(tmp_dir + '%s.png' % index)
    width, height = fromImage.size
    toImage = Image.new('RGB', (width * W, height * H))
    for i in range(H):
        for j in range(W):
            fname = tmp_dir + '%s.png' % index
            try:
                fromImage = Image.open(fname) if index >= 0 else Image.new('RGB', (width, height), 'white')
            except Exception as e:
                fromImage = Image.new('RGB', (width, height), 'white')

            toImage.paste(fromImage, (j * width, i * height))
            index += 1

    for i in range(W * H):
        fname = tmp_dir + '%s.png' % i
        if os.path.exists(fname):
            os.remove(fname)
    toImage.save(tgt)
    return doc.pageCount


def paper_to_image(pdf_file_name: str) -> str:
    """ Convert a paper PDF to an image with a 2 x 4 panel """
    images = convert_from_path(pdf_file_name, 200, None, None, None, 'jpg', None, 1)
    num_page = len(images)
    if num_page > 6:
        # Create an empty image with a 2 x 4 pages panel
        total_width = 0
        max_height = 0

        new_width = round(images[0].width * 4)
        new_height = round(images[0].height * 2)

        new_im = Image.new('RGB', (new_width, new_height), (255, 255, 255))

        # Copy and paste pages from pages 1-4
        x_offset = 0
        y_offset = 0
        for i in range(4):
            new_im.paste(images[i], (x_offset, y_offset))
            x_offset += images[i].size[0]

        # Copy and paste pages from pages 5-8
        x_offset = 0
        y_offset += images[i].size[1]

        for i in range(4, 8):
            if i < num_page:
                new_im.paste(images[i], (x_offset, y_offset))
                x_offset += images[i].size[0]

    else:
        BasicException('We process PDF with at least 7 pages long.')

    # Save the image as a JPG
    img_file_name = pdf_file_name[0:-4] + '.jpg'
    new_im = new_im.resize((3400, 2200))
    new_im.save(img_file_name)

    # Resize the image to [680, 440] and remove header (to avoid data leakage)
    img = cv2.imread(img_file_name)
    img = cv2.resize(img, dsize=(680, 440), interpolation=cv2.INTER_AREA)
    img[0:15, 0:150] = 255  # remove header (to avoid data leakage)
    cv2.imwrite(img_file_name, img)

    print('Converted the PDF ' + pdf_file_name)

    return img_file_name


def save_pages(src, tgt):
    doc = fitz.open(src)
    # if doc.pageCount < W * H + 1: # pagenum>=W*H without considering reference page
    #     return

    for pg in range(doc.pageCount):  # save every page as an image
        page = doc[pg]
        zoom = 40
        rotate = 0
        trans = fitz.Matrix(zoom / 100.0, zoom / 100.0).preRotate(rotate)

        # create raster image of page (non-transparent)
        pm = page.getPixmap(matrix=trans, alpha=False)

        # write a PNG image of the page
        pm.writePNG(tgt + '%s.png' % pg)

    return doc.pageCount


if __name__ == '__main__':
    print('execute nn_process.py ...')
    conf_fold = INPUT_DIR + 'train/conference/'
    arxiv_fold = INPUT_DIR + 'train/workshop/'
    tmp_dir = INPUT_DIR + 'temp/'

    # make folders
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for mode in ['train', 'test']:
        for name in ['conference', 'workshop']:
            fold = TGT_INPUT_DIR+'/%s/%s/' % (mode, name)
            if not os.path.exists(fold):
                os.makedirs(fold)

    # convert pdf to images
    for fold, s in [[conf_fold, 'conference'], [arxiv_fold, 'workshop']]:
        tgt_fold = TGT_INPUT_DIR+'train/%s/' % s
        for pdf in tqdm(os.listdir(fold)):
            # src: path of pdf
            # tgt: path of img
            src = fold + pdf
            tgt = tgt_fold + pdf.replace('pdf', 'jpg')  # complete paper
            # tgt = tgt_fold + pdf.replace('.pdf', '')  # pages
            try:
                save_jpg(src, tgt, tmp_dir)  # overall paper
                # save_pages(src, tgt)
            except:
                pass

    # train_test_split
    arxiv_lst = os.listdir(TGT_INPUT_DIR+'train/workshop')
    conf_lst = os.listdir(TGT_INPUT_DIR+'train/conference')
    random.shuffle(arxiv_lst)
    random.shuffle(conf_lst)
    for i in arxiv_lst[:int(len(arxiv_lst) * TEST_SIZE)]:
        shutil.move(TGT_INPUT_DIR+'train/workshop/' + i, TGT_INPUT_DIR+'test/workshop/' + i)
    for i in conf_lst[:int(len(conf_lst) * TEST_SIZE)]:
        shutil.move(TGT_INPUT_DIR+'train/conference/' + i, TGT_INPUT_DIR+'test/conference/' + i)