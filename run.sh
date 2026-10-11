#! /bin/bash


source ~/ml_env/bin/activate

python MNIST_multi_test.py configs/gen_05/config_22.json
python MNIST_multi_test.py configs/gen_05/config_23.json
# python MNIST_multi_test.py configs/gen_05/config_24.json
# python MNIST_multi_test.py configs/gen_05/config_25.json

ga
gc "big run test"
gp
