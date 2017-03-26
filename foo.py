import tensorflow as tf
from src.Dataset import *

d = load_FlyingChairs('/home/mtesfald/Datasets/FlyingChairs/FlyingChairs_release/data')
X, y = d.next_batch(10)
