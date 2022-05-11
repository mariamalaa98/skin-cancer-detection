from torch.optim import Adam
import torch
import time
import sys


def train(model, epochs, train_loader, lr, cuda=True):
    optimizer = Adam(model.parameters(), lr)
    loss_function = torch.nn.BCELoss()
    batch_size = train_loader.batch_size
    no_batches = len(train_loader)
    dataset_size = float(len(train_loader.dataset))
    train_losses=[]

    if cuda:
        model.cuda()

    for e in range(epochs):
        loss_sum = 0.0
        cnt = 0.0
        time_sum = 0.0
        for img, value in train_loader:
            ts = time.time()
            optimizer.zero_grad()
            if cuda:
                img = img.cuda()
                value = value.cuda()

            output = model(img)
            loss = loss_function(output, value)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * batch_size
            cnt += 1.0
            finished = int((cnt * 10) / no_batches)
            remaining = 10 - finished
            te = time.time()
            time_sum += (te - ts)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)
            sys.stdout.flush()
            sys.stdout.write("\r epoch " + str(e + 1) + " [" + str(
                "=" * int((cnt * 10) / no_batches) + str("." * remaining) + "] time remaining = " + str(
                    time_remaing / 60.0)[:8]))
        epoch_loss = loss_sum / dataset_size
        train_losses.append(epoch_loss)

        print(f" epoch {e + 1} loss ={epoch_loss}")
    return train_losses
