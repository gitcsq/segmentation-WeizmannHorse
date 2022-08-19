import os.path
import numpy as np
from model_unet_plus import NestedUNet as unet_plus
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import dataset
import torch
from pathlib import Path
np.random.seed(1)
from trainer import train_one_epoch


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    model = unet_plus()
    device = torch.device('cuda')
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = dataset.horse_dataset(transform=transform, img_size=args.img_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    for epoch in range(args.epoch):
        train_one_epoch(args=args, epoch=epoch, model=model, dataloader=train_dataloader,
                        optimizer=optimizer, criterion=criterion)

        epoch_name = str(epoch)
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epoch:
            checkpoint_paths = [Path(args.output_dir) / ('depth%s-ep%s.pth' % (args.depth, epoch_name))]
            for checkpoint_path in checkpoint_paths:
                to_save = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                torch.save(to_save, checkpoint_path)

