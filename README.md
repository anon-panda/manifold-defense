# mainfold-defense
Code for NeurIPS submission 8302

To download source code and files for the DefenseGAN break, go to 
https://drive.google.com/file/d/15rpmyaq6clZBMHbXvjs5CrdrY2lJpOA3/view?usp=sharing

To run the DefenseGAN break experiments, run the file make_intermediates.py in the folder break/


To train a robust madry classifier (checkpoint is provided in the repo folder results/), run:

python adv_train.py --num-pgd 40 --mode l2 --eps 1.5 --pgd-lr 0.3 --no-norm --no-sn --random-step --net good_7 --baseline --with-decode --dataset mnist --op-attack --save-str madry_1.5_40_0.3_op_5iter_validation_2 --embed-feats 256 --opt adam --sgd-lr 1e-4 --resume madry_mnist_1.5_40 --validation-set --op-generator trained_vae_leakyrelu_20_500_500_784.pth --op-embed-feats 20 

Please also clone the repo in
https://github.com/JianGoForIt/YellowFin_Pytorch.git
