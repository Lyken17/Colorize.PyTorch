import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
from model import TransformerNet


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 12, 'pin_memory': False}
    else:
        kwargs = {}

    class RGB2YUV(object):
        def __call__(self, img):
            import numpy as np
            import cv2

            npimg = np.array(img)
            yuvnpimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2YUV)
            pilimg = Image.fromarray(yuvnpimg)

            return pilimg

    from transforms.color_space import Linearize, SRGB2XYZ, XYZ2CIE

    SRGB2LMS = transforms.Compose([
        Linearize(),
        SRGB2XYZ(),
        XYZ2CIE(),
    ])

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        SRGB2LMS
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformer = TransformerNet(in_channels=2, out_channels=1)  # input: L S, predict: M
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    # vgg = Vgg16()
    # utils.init_vgg16(args.vgg_model_dir)
    # vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    transformer = nn.DataParallel(transformer)

    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested, but related driver/device is not set properly.")
        transformer.cuda()

    for e in range(args.epochs):
        # initialization
        transformer.train()
        count = 0
        moving_loss = 0.0

        # Train one epoch
        for batch_id, (imgs, _) in enumerate(train_loader):
            n_batch = len(imgs)
            count += n_batch
            optimizer.zero_grad()
            # L & S channel
            x = torch.cat([imgs[:, :1, :, :].clone(), imgs[:, -1:, :, :].clone()], dim=1)
            # M channels
            gt = imgs[:, 1:2, :, :].clone()

            # print(x.size(), gt.size())
            if args.cuda:
                x = x.cuda()
                gt = gt.cuda()

            y = transformer(x)

            total_loss = mse_loss(y, gt)
            total_loss.backward()
            optimizer.step()

            moving_loss = moving_loss * 0.9 + total_loss.item() * 0.1

            if batch_id % args.log_interval == 0:
                msg = "{} | Epoch {}:\t[{}/{}]\ttotal: {:.6f} ({:.6f})".format(
                    time.ctime(), e + 1, batch_id, len(train_loader),
                    total_loss.item(), moving_loss)
                print(msg)

        # Evaluation and save model
        transformer.eval()
        # transformer.cpu()
        save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') \
                              + "_" + str("%.6f" % moving_loss) + ".model"
        os.makedirs(args.save_model_dir, exist_ok=True)
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        # if not os.path.exists(args.vgg_model_dir):
        #     os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def stylize(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.content_scale)
    content_image = content_image.unsqueeze(0)

    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))

    if args.cuda:
        style_model.cuda()

    output = style_model(content_image)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=30,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    # train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
    #                               help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, default="ckpt",
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--style-weight", type=float, default=5.0,
                                  help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=10,
                                  help="number of images after which the training loss is logged, default is 500")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
