
import json
import datetime
from copy import deepcopy
import sys
import statistics

import torch

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from model import *
from plotting import *


CFG_NAME = sys.argv[1]
SHOW_FIGS = False
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
START_SEED = CFG["seed"]
USE_N_SEEDS = CFG["use_n_seeds"]

# get data
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(float) / 255.0   # normalize to [0, 1]
y_all = mnist.target.astype(int)

log(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"), False)
log(f'Dataset shape: X={X_all.shape}, y={y_all.shape}', False)
log(print_to_screen=False)

# Split dataset
TEST_SIZE, VAL_SIZE = CFG["dataset"]["test_split"], CFG["dataset"]["valid_split"]


# aggregate results
l_model_names = [i["name"] for i in CFG["models"]]
d_accuracy = {name: list() for name in l_model_names}
d_test_loss = {name: list() for name in l_model_names}
d_best_valid_loss = {name: list() for name in l_model_names}
d_best_train_loss = {name: list() for name in l_model_names}
d_best_epoch = {name: list() for name in l_model_names}


for SEED in range(START_SEED, START_SEED + USE_N_SEEDS):

    log(f"\nUsing seed {SEED}\n", False)

    # Split: 70 / 15 / 15
    X_temp, X_test, y_temp, Y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=SEED)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED)

    # make tensors
    d_dtypes = {
        32: torch.float32,
        64: torch.float64
    }
    dtype = d_dtypes[CFG["dataset"].get("bits", 32)]
    label_dtype = torch.long
    t_X_train = torch.tensor(X_train, dtype=dtype)
    t_X_valid = torch.tensor(X_valid, dtype=dtype)
    t_X_test = torch.tensor(X_test, dtype=dtype)
    t_Y_train = torch.tensor(Y_train, dtype=label_dtype)
    t_Y_valid = torch.tensor(Y_valid, dtype=label_dtype)
    t_Y_test = torch.tensor(Y_test, dtype=label_dtype)

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
    

    # TRAIN
    EPOCHS = CFG["training"]["epochs"]
    BATCH_SIZE = CFG["training"]["batch_size"]
    LR = CFG["training"]["lr"]
    loss_func = torch.nn.CrossEntropyLoss()

    for model_name, model, initializer in list(zip(l_model_names, l_all_models, l_initializers)):

        # initialize model params
        initializer(model)

        # log sample weight matrix
        log(f"{model_name}", False)
        log(f"Sample weight shape: {model.layers[0].weight.shape}", False)
        log(f"Kaiming variance: {2 / sum(model.layers[0].weight.shape)}", False)
        log(f"Matrix variance: {torch.var(model.layers[0].weight, unbiased=False)}", False)
        log(f"Mean row variance (feature-to-feature weight variance): {torch.mean(torch.var(model.layers[0].weight, 0, unbiased=False))}", False)
        log(f"Mean col variance (neuron-to-neuron weight variance): {torch.mean(torch.var(model.layers[0].weight, 1, unbiased=False))}", False)

        optim = torch.optim.AdamW(model.parameters(), lr=LR)
        validation_losses = []
        train_losses = []
        
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
            plot_loss(train_losses, validation_losses)

        # predict test set
        t_Yhat = model(t_X_test)
        loss_test = loss_func(t_Yhat, t_Y_test) 

        # acc
        correct = (torch.argmax(t_Yhat, dim=-1) == t_Y_test).sum()
        total = t_Yhat.shape[0]
        acc = correct / total

        # log
        log(f"Accuracy: {acc}", False)
        log(f"Best train loss: {min(train_losses)}", False)
        log(f"Best validation loss: {min(validation_losses)}", False)
        log(f"Test loss: {loss_test.item()}", False)
        log(f"Best model epoch: {best_model_epoch}", False)
        log(print_to_screen=False)

        # aggregate stats
        d_accuracy[model_name].append(acc.item())
        d_test_loss[model_name].append(loss_test.item())
        d_best_valid_loss[model_name].append(min(validation_losses))
        d_best_train_loss[model_name].append(min(train_losses))
        d_best_epoch[model_name].append(best_model_epoch)

    print(f"Seed {SEED - START_SEED + 1} / {USE_N_SEEDS} complete")

# average dicts
d_accuracy = {key: {"mean": round(sum(val) / len(val), 5), "std": round(statistics.stdev(val), 5)} for (key, val) in d_accuracy.items()}
d_test_loss = {key: {"mean": round(sum(val) / len(val), 5), "std": round(statistics.stdev(val), 5)} for (key, val) in d_test_loss.items()}
d_best_valid_loss = {key: {"mean": round(sum(val) / len(val), 5), "std": round(statistics.stdev(val), 5)} for (key, val) in d_best_valid_loss.items()}
d_best_train_loss = {key: {"mean": round(sum(val) / len(val), 5), "std": round(statistics.stdev(val), 5)} for (key, val) in d_best_train_loss.items()}
d_best_epoch = {key: {"mean": round(sum(val) / len(val), 5), "std": round(statistics.stdev(val), 5)} for (key, val) in d_best_epoch.items()}

log(f"\n\n\nAggregate results")
log(f"Accuracy: {json.dumps(d_accuracy, indent=4)}")
log(f"Test loss: {json.dumps(d_test_loss, indent=4)}")
log(f"Best validation loss: {json.dumps(d_best_valid_loss, indent=4)}")
log(f"Best train loss: {json.dumps(d_best_train_loss, indent=4)}")
log(f"Best epoch: {json.dumps(d_best_epoch, indent=4)}")
    
