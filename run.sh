#! /bin/bash

source ~/ml_env/bin/activate

python MNIST_multi_test.py configs/gen_06/config_26.json
python MNIST_multi_test.py configs/gen_06/config_27.json
python MNIST_multi_test.py configs/gen_06/config_28.json
python MNIST_multi_test.py configs/gen_06/config_29.json

git add .
git commit -m"big run (IntraNVG)"
git push
