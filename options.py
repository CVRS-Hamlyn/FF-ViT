from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(os.getcwd()) # the directory that options.py resides in

print(file_dir)
data_folder = os.path.join(file_dir, 'Your/Data/Path')
model_folder = os.path.join(file_dir, 'Your/ckpt/path')
class AutoFocusOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="AutoFocus options")

        # PATHS
        self.parser.add_argument("--root_path",
                                 type=str,
                                 help="The root path",
                                 default=os.path.join(data_folder))
        self.parser.add_argument("--model_folder",
                                 type=str,
                                 help="The folder for storing models",
                                 default=model_folder)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="log_book")
        self.parser.add_argument("--checkpoint_dir",
                                 type=str,
                                 help="The checkpoint directory",
                                 default='freq_attn_RBF')
        
        # DATA FEATURES
        self.parser.add_argument("--width",
                                 type=int,
                                 help="width of pCLE image",
                                 default=448)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="height of pCLE image",
                                 default=256)
        self.parser.add_argument("--width_p",
                                 type=int,
                                 help="width of the patch",
                                 default=56)
        self.parser.add_argument("--height_p",
                                 type=int,
                                 help="height of the patch",
                                 default=64)
        self.parser.add_argument("--k",
                                 type=int,
                                 help="No.k fold dataset",
                                 default=None)
        # TRAINING options
        self.parser.add_argument("--in_channels",
				                 type=int,
				                 help="number of input channels",
				                 default=1)
        self.parser.add_argument("--out_channels",
                                 type=int,
                                 help="number of output channels",
                                 default=1)
        self.parser.add_argument("--hidden_dim",
                                 type=int,
                                 help="the hidden dimensions of the model",
                                 default=384)
        self.parser.add_argument("--depth",
                                 type=int,
                                 help="the number of self-attention layers of ViT",
                                 default=12)
        self.parser.add_argument("--heads",
                                 type=int,
                                 help="the number of heads in self-attention layers",
                                 default=8)
        self.parser.add_argument("--mlp_dim",
                                 type=int,
                                 help="the dimensions of hidden features at MLP",
                                 default=1536)
        self.parser.add_argument("--no_ffpe",
                                 help="whether use fast Fourier patch embedding",
                                 action="store_true")
        self.parser.add_argument("--no_cross_attn",
                                 help="whether use cross attention between blur metric map and latent representations",
                                 action="store_true")      
        self.parser.add_argument("--no_msi",
                                 help="whether use multi-scale inference",
                                 action="store_true")   
        self.parser.add_argument("--pyr_decay",
                                 type=float,
                                 help="decay coefficient for Multi-scale inference",
                                 default=0.5)             
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="The type of model",
                                 default="FF-ViT")
        self.parser.add_argument("--shift",
                                 help="whether shift the input image",
                                 action="store_true")
        self.parser.add_argument("--num_pyrs",
                                 type=int,
                                 help="The number of pyramids",
                                 default=3)
        self.parser.add_argument("--self_locality",
                                 help="whether apply LSA to self-attention layer",
                                 action="store_true")
        self.parser.add_argument("--cross_locality",
                                 help="whether apply LSA to cross-attention layer",
                                 action="store_true")


        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=128)
        self.parser.add_argument("--optim",
                                 type=str,
                                 help="optimizer",
                                 default="adamw")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="regularization",
                                 default=1e-2)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=100)
        self.parser.add_argument("--l1_weight",
                                 type=float,
                                 help="the weight of MAE to balance the loss function",
                                 default=1e-1)
        self.parser.add_argument("--lamda",
                                 type=float,
                                 help="The weight to balance the contribution",
                                 default=1e-3)
        


        # ABLATION options
        self.parser.add_argument("--pretrained",
                                 help="if set use pretrained weights",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)
        self.parser.add_argument("--multi_gpu",
                                 help="if set use multi_gpu",
                                 action="store_true")
        self.parser.add_argument("--gpu_id",
                                 type=int,
                                 help="Tthe id of gpu to be used",
                                 default=0)
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


