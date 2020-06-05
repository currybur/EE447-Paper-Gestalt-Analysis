import csv
from nn_process import save_pages
from predict import predict_page, get_nn_model
import os

DATASET_DIR = '../dataset_new/'
TMP_DIR = '../input/temp/'

def extract_save_scores(source_path, writer, label):
    pdf_name = source_path.split('/')[-1]
    page_num = save_pages(source_path, TMP_DIR+pdf_name.replace('.pdf',''))
    scores = []
    for p in os.listdir(TMP_DIR):
        p = TMP_DIR + p
        s = predict_page(p, model)
        scores.append(s)

    for pt in os.listdir(TMP_DIR):
        pt = TMP_DIR + pt
        if os.path.exists(pt):
            os.remove(pt)

    score_string = ','.join(str(i) for i in scores)
    try:
        writer.writerow([pdf_name, page_num, score_string, label])
        return True
    except:
        return false


if __name__ == '__main__':
    saved_file = open('scores.csv','w',encoding='utf-8',newline='')
    csv_writer = csv.writer(saved_file)
    csv_writer.writerow(['title','pages','scores','label'])
    model = get_nn_model()

    for t in ["train","test"]:
        for c, l  in [["conference",1],["workshop",0]]:
            source_path = DATASET_DIR + t +'/' + c + '/'
            num = len(os.listdir(source_path))
            print("totally {} file in {}".format(num, t+'/'+c))
            count = 1
            for p in os.listdir(source_path):
                try:
                    p = DATASET_DIR+t +'/' + c + '/' + p
                    extract_save_scores(p, csv_writer, l)
                    print("processing file {} {}/{}".format(p, count, num))
                    count += 1
                except:
                    pass

    saved_file.close()
            