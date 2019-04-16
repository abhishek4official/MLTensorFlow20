from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import pandas as pd
import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.gridspec as gridspec 
#%matplotlib auto

import tensorflow as tf
import os




# This module defines the show_graph() function to visualize a TensorFlow graph within Jupyter.

# As far as I can tell, this code was originally written by Alex Mordvintsev at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

# The original code only worked on Chrome (because of the use of <link rel="import"...>, but the version below
# uses Polyfill (copied from this StackOverflow answer: https://stackoverflow.com/a/41463991/38626)
# so that it can work on other browsers as well.


from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
#Reset Graph
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def main():
    print("Hello World!")
    # Load Data
    creditData = pd.read_csv("/home/vicky/abhishek/StocksAI/Hands-on-Deep-Learning-with-TensorFlow-2.0/Section 1/dataset/ChurnBank.csv",encoding="utf-8",index_col=0)
    # Data print Header
    print(creditData.head())
    # print Data Describe
    print(creditData.describe())
    #Print col Names
    creditCols = list(creditData.columns)
    print(creditCols)
    creditData[['SEX', 'EDUCATION', 'MARRIAGE','default payment next month']].hist(figsize=(15,15))
    plt.show()
    print(plt.get_backend())
    categoricalCols = ["AGE_RANGE","SEX","EDUCATION","MARRIAGE"]
    continuousCols = ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    # Change 2=0 in sex column
    creditData.loc[creditData["SEX"]==2,"SEX"] = 0
    

if __name__== "__main__":
    main()
# Load Data
# if __name__== "__main__":
# creditData = pd.read_csv("/home/vicky/abhishek/StocksAI/Hands-on-Deep-Learning-with-TensorFlow-2.0/Section 1/dataset/ChurnBank.csv",encoding="utf-8",index_col=0)
# creditData.head()