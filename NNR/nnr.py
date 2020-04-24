import sys
import os
sys.path.append(os.pardir)
from dataset.fpga_fp_data import generate_data
import matplotlib.pyplot as plt              
import numpy as np
import pandas as pd
from statistics import mean
import pickle
import generate_nnr_data

residual_data, aged_residual_data = generate_nnr_data.generate_nnr()

