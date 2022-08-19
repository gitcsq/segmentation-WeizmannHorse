import torch


def train_one_epoch(args, epoch, model, dataloader, optimizer, criterion):
    device = torch.device('cuda')
    loss_in_one_epoch = 0
    for batch_idx, data in enumerate(dataloader):
        x, y = data
        x = x.type(torch.float32)
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_in_one_epoch += loss_value
    loss_in_one_epoch = loss_in_one_epoch/int(4000/args.batch_size)
    print('epoch:{}/{}  loss:{}  '.format(epoch, args.epoch, loss_in_one_epoch))

