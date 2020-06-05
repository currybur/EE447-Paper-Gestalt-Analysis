def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the paper gestalt model
# (pre-trained on CVPR/ICCV 2013-2017 conference/workshop papers)
gestalt_model = models.resnet18(pretrained=False)
gestalt_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
gestalt_model.fc = nn.Linear(gestalt_model.fc.in_features, 2)
gestalt_model.load_state_dict(torch.load('../output/nn_output/aperNet.pth'))
gestalt_model = gestalt_model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
file_name = "upload.pdf"
img_file_name = paper_to_image(file_name)

# Temporature
T = 5

with torch.no_grad():
  im = Image.open(img_file_name)
  im_tensor = preprocess(im)
  im_tensor = im_tensor.to(device)
  im_tensor = im_tensor.unsqueeze(0)

  pred_logit = gestalt_model(im_tensor)
  pred_logit = pred_logit.to(torch.device("cpu")).numpy() / T

  pred_prob = softmax(pred_logit[0])

  print('Our classifier predicts Pr[Good paper|{paper_pdf}] = {prob:.02f}% '\
        .format(paper_pdf= file_name, prob = pred_prob[0]*100))