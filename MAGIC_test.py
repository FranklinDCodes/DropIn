
import json
import datetime
import requests
from io import StringIO
from copy import deepcopy
import sys

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib

from model import *
from plotting import *


CFG_NAME = sys.argv[1]
SHOW_FIGS = True
with open(CFG_NAME, 'r') as fl:
	CFG = json.load(fl) 
	
LOG_NAME = f"logs/{CFG_NAME.split('.')[-2].split('/')[-1]}_{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log" 

def log(s: any = "", print_to_screen: bool = True, write_to_file: bool = True) -> None: 
	
	if print_to_screen: 
		print(s) 

	if write_to_file:
		with open(LOG_NAME, 'a') as fl:
			fl.write(str(s) + '\n')


# seed
SEED = CFG["seed"]
np.random.seed(SEED)
torch.random.manual_seed(SEED)


# get data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
response = requests.get(url)
response.raise_for_status()
columns = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
    "class"
]
df = pd.read_csv(
    StringIO(response.text),
    header=None,
    names=columns
)

# setup labels
df['label'] = (df['class'] == 'g').astype(int)

# setup dataset
features = ['fLength','fWidth','fSize','fConc','fConc1',
            'fAsym','fM3Long','fM3Trans','fAlpha','fDist']
X = df[features].values          # shape (19020, 10)
y = df['label'].values.reshape(-1, 1)  # shape (19020, 1)


# Split dataset
TEST_SIZE, VAL_SIZE = CFG["dataset"]["test_split"], CFG["dataset"]["valid_split"]
X_train_and_valid, X_test, Y_train_and_valid, Y_test  = train_test_split(X, y, test_size=TEST_SIZE)
X_train, X_valid, Y_train, Y_valid  = train_test_split(X_train_and_valid, Y_train_and_valid, test_size=VAL_SIZE / (1 - TEST_SIZE)) # take VAL_SIZE of the original dataset size

norm = MinMaxScaler()
X_train = norm.fit_transform(X_train)
X_valid = norm.transform(X_valid)
X_test = norm.transform(X_test)

# make tensors
d_dtypes = {
	32: torch.float32,
	64: torch.float64
}
dtype = d_dtypes[CFG["dataset"].get("bits", 32)]
t_X_train = torch.tensor(X_train, dtype=dtype)
t_X_valid = torch.tensor(X_valid, dtype=dtype)
t_X_test = torch.tensor(X_test, dtype=dtype)
t_Y_train = torch.tensor(Y_train, dtype=dtype)
t_Y_valid = torch.tensor(Y_valid, dtype=dtype)
t_Y_test = torch.tensor(Y_test, dtype=dtype)


# create models
l_all_models = list()
l_model_names = list()
l_initializers = list()
l_model_cfgs = CFG["models"]
for model_cfg in l_model_cfgs:

    # unpack model config
    use_drop_in = model_cfg.get("drop_in", False)
    stop_drop_in_after = model_cfg.get("stop_drop_in_after", None)
    drop_in_kwargs = model_cfg.get("drop_in_kwargs", dict())
    layers = model_cfg["layers"]
    activation = model_cfg.get("activation", "Tanh")

    # make model
    model = Model(
        t_X_train.shape[-1],
        layers,
        use_drop_in,
        stop_drop_in_after,
        drop_in_kwargs,
        activation
    )

    init_cfg = model_cfg.get("init", {"type": "glorot"})
    init_type = init_cfg["type"]
    init_kwargs = init_cfg.get("kwargs", dict())

    initializer = ParamInitializer(SEED, init_type, **init_kwargs)

    l_all_models.append(model)
    l_model_names.append(model_cfg["name"])
    l_initializers.append(initializer)


# start logging
log(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
log()
log(f'Dataset shape: X={X.shape}, y={y.shape}')
log(f'Class balance: {y.mean():.3f} gamma, {1-y.mean():.3f} hadron')
log()


# TRAIN
EPOCHS = CFG["training"]["epochs"]
BATCH_SIZE = CFG["training"]["batch_size"]
LR = CFG["training"]["lr"]
loss_func = torch.nn.BCEWithLogitsLoss()

for model_name, model, initializer in list(zip(l_model_names, l_all_models, l_initializers)):

    print(f"Starting {model_name}...")

    # initialize model params
    initializer(model)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    train_losses = []
    validation_losses = []
	
    best_val_loss = float("inf")
    best_model = deepcopy(model)
    best_model_epoch = 0
	
    for epoch in range(EPOCHS):

        # shuffle
        t_shuffle_indices = torch.randperm(t_X_train.shape[0])
        t_X_train_shuffled = t_X_train[t_shuffle_indices, :]
        t_Y_train_shuffled = t_Y_train[t_shuffle_indices]

        # split batches
        t_X_train_batched = torch.split(t_X_train_shuffled, BATCH_SIZE)
        t_Y_train_batched = torch.split(t_Y_train_shuffled, BATCH_SIZE)

        epoch_losses = []

        # iterate
        for batch_idx in range(len(t_X_train_batched)):

            # unpack batch
            batch_x = t_X_train_batched[batch_idx]
            batch_y = t_Y_train_batched[batch_idx]
            
            # forward pass model
            t_Yhat = model(batch_x)

            # calculate CE loss
            loss = loss_func(t_Yhat, batch_y)
            epoch_losses.append(loss.item() * batch_y.shape[0])

            # optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

        # add average epoch loss to history
        # inter-epoch reduction strategy of weighted averaging suggested by ChatGPT
        train_losses.append(sum(epoch_losses) / X_train.shape[0]) 

        # validation 
        # forward pass model 
        t_Yhat = model(t_X_valid) 
        loss_validation = loss_func(t_Yhat, t_Y_valid) 
        validation_losses.append(loss_validation.item())

        # save model if new is best
        if loss_validation < best_val_loss:
            best_model = deepcopy(model)
            best_val_loss = loss_validation
            best_model_epoch = epoch

        model.update(epoch + 1)

    if SHOW_FIGS:
        matplotlib.use('TkAgg')
        plot_loss(train_losses, validation_losses)

    # predict test set
    t_Yhat = model(t_X_test)
    y_hat_test_classes = (t_Yhat >= 0.5)
    loss_test = loss_func(t_Yhat, t_Y_test) 

    # acc
    y_hat_test_classes = y_hat_test_classes.squeeze()
    correct = (y_hat_test_classes == torch.squeeze(t_Y_test)).sum()
    total = y_hat_test_classes.shape[0]
    acc = correct / total
    fpr, tpr, _ = roc_curve(np.squeeze(t_Y_test.numpy()), np.squeeze(t_Yhat.detach().numpy()))
    roc_auc = auc(fpr, tpr)

    if SHOW_FIGS:

        plot_confusion_matrix_labeled(torch.squeeze(t_Y_test), y_hat_test_classes, ["Hadron", "Gamma"])

        # plot code borrowed and modified from GFG
        # https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
        plt.figure(figsize=(7, 5))

        plt.plot(fpr, tpr, label=f'Neural Net (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    # log
    log(f"{model_name}")
    log(f"Accuracy: {acc}")
    log(f"Test loss: {loss_test.item()}")
    log(f"AUC: {roc_auc}")
    log(f"Best model epoch: {best_model_epoch}")

    log()



