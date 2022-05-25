from datetime import datetime

from torch.optim import Adam
import torch
import time
import sys
import os
import pandas as pd

from torch.optim import Adam
import time


def train(model, epochs, train_loader, lr, cuda=True):
    optimizer = Adam(model.parameters(), lr)
    loss_function = torch.nn.BCELoss()
    batch_size = train_loader.batch_size
    no_batches = len(train_loader)
    dataset_size = float(len(train_loader.dataset))
    train_losses = []

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

        # TODO add testing loop and calculate test loss and add it to array of test losses []
        # TODO save model weight if both train loss and test loss are below the minimum they had reached

        print(f" epoch {e + 1} loss ={epoch_loss}")
        # TODO return array of train_losses and test_losses
    return train_losses


def save_epochs_to_csv(csv_save_path, train_loss, no_train_rows, test_loss, no_test_rows, time_taken, notes=None):
    if notes is None:
        notes = ""
    date_now = datetime.now()
    if len(csv_save_path) == 0:
        full_path = "train_data.csv"
    else:
        full_path = f"{csv_save_path}/train_data.csv"
    row = [[train_loss, no_train_rows, test_loss, no_test_rows, time_taken, notes, date_now.strftime('%d/%m/%Y'),
            date_now.strftime('%I:%M %p')]]
    df = pd.DataFrame(row,
                      columns=["Train Loss", "no train rows", "Test Loss", "No test rows", "Time taken (M)", "Notes",
                               "Date", "Time"])

    if not os.path.exists(full_path):
        df.to_csv(full_path)
    else:
        df.to_csv(full_path, mode='a', header=False)


def classifier_train_step(model, optimizer, loss_function, img, data, label, cuda):
    optimizer.zero_grad()
    if cuda:
        img = img.cuda()
        data = data.cuda()
        label = label.cuda()
    output = model(img, data)
    loss = loss_function(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()


def features_train_step(model, optimizer, loss_function, img, label, cuda):
    optimizer.zero_grad()
    if cuda:
        img = img.cuda()
        label = label.cuda()
    output = model(img)
    loss = loss_function(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()


def classifier_test_step(model, loss_function, img, data, label, cuda):
    if cuda:
        img = img.cuda()
        label = label.cuda()
        data = data.cuda()
    output = model(img, data)
    loss = loss_function(output, label)

    return loss.item()


def features_test_step(model, loss_function, img, label, cuda):
    if cuda:
        img = img.cuda()
        label = label.cuda()
    output = model(img)
    loss = loss_function(output, label)

    return loss.item()


def test_model(model, test_loader, loss_function, cuda, test_type):
    batch_size = test_loader.batch_size
    no_batches = len(test_loader)
    dataset_size = float(len(test_loader.dataset))
    model.eval()
    if cuda:
        model.cuda()
    loss_sum = 0.0
    cnt = 0.0
    time_sum = 0.0
    with torch.no_grad():
        for data in test_loader:
            ts = time.time()

            if test_type == "features":
                img, label = data
                loss = features_test_step(model, loss_function, img, label, cuda)
            elif test_type == "classifier":
                img, data, label = data
                loss = classifier_test_step(model, loss_function, img, data, label, cuda)
            else:
                raise ValueError("invalid test mod")

            loss_sum += loss * test_loader.batch_size

            cnt += 1.0
            finished = int((cnt * 10) / no_batches)
            remaining = 10 - finished
            te = time.time()
            time_sum += (te - ts)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)
            sys.stdout.write("\r Testing  [" + str(
                "=" * finished + str("." * remaining) + "] time remaining = " + str(
                    time_remaing / 60.0)[:8] + " Avg Test_Loss=" + str(loss_sum / (cnt * batch_size))[:8]))
    loss_sum /= dataset_size
    return loss_sum


def train_model(model, no_of_epochs, lr, train_loader, test_loader, train_type, cuda=False, weight_saving_path=None,
                epoch_data_saving_path=None, notes=None):
    train_losses_list = []
    test_losses_list = []
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.BCELoss()
    no_batches = len(train_loader)
    dataset_size = float(len(train_loader.dataset))
    min_test_loss = test_model(model, test_loader, loss_function, cuda, train_type)
    model.cuda()

    print(f"Test loss before Training {min_test_loss}")
    for e in range(no_of_epochs):
        model.train()
        train_loss_sum = 0.0
        cnt = 0.0
        time_sum = 0.0

        epoch_time_start = time.time()

        for data_row in train_loader:
            ts = time.time()

            if train_type == "features":
                img, label = data_row
                loss = features_train_step(model, optimizer, loss_function, img, label, cuda)
            elif train_type == "classifier":
                img, data, label = data_row
                loss = classifier_train_step(model, optimizer, loss_function, img, data, label, cuda)
            else:
                raise ValueError("invalid train mod")
            current_train_loss = loss* train_loader.batch_size
            train_loss_sum += current_train_loss

            # calculate epoch info
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
                    time_remaing / 60.0)[:8]) + " Avg train_loss=" + str(
                train_loss_sum / (cnt * train_loader.batch_size))[:8])
        print()
        train_loss_avg = train_loss_sum / dataset_size
        train_losses_list.append(train_loss_avg)

        test_loss = test_model(model, test_loader, loss_function, cuda, train_type)
        test_losses_list.append(test_loss)
        print()
        if test_loss < min_test_loss:
            save_train_weights(model, train_loss_avg, min_test_loss, weight_saving_path)
            min_test_loss = test_loss
            print(f"new train loss ={train_loss_avg} new test loss= {test_loss}")

        if test_loss > train_loss_avg:
            print(" over fitting")
        epoch_time_end = time.time()
        epoch_time = f"{(epoch_time_end - epoch_time_start) / 60}:00"

        save_epochs_to_csv(epoch_data_saving_path, train_loss_avg, len(train_loader.dataset), test_loss,
                           len(test_loader.dataset), epoch_time, notes)

        print(f" epoch {e + 1} train_loss ={train_loss_avg} test_loss={test_loss}")


def save_weights(model, path):
    state_dict = model.state_dict()
    torch.save(state_dict, path)


def save_train_weights(model, train_loss, test_loss, saving_path):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]}) Test_({str(test_loss)[:8]}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path
