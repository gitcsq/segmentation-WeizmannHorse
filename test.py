from model_unet_plus import NestedUNet as unet_plus
from evaluation_functions import evaluation
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import dataset
import torch


def prepare_model(chkpt_dir, hori=False):
    # build model
    depth = 5
    model = unet_plus()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=torch.device('cpu'))
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def main(args):

    with torch.no_grad():
        device = torch.device('cpu')
        model = prepare_model(args.checkpoint_path)
        model = model.to(device)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        train_dataset = dataset.horse_dataset(transform=transform, mode='test')
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=100)
        for i, data in enumerate(train_dataloader):
            x, y, name = data
            x = x.type(torch.float32)
            x = x.to(device)
            y_pred = model(x)
            y = y.unsqueeze(1)
            y_pred = torch.max(y_pred, 1)[1].unsqueeze(1)
            evaluation(y_pred, y, x)



