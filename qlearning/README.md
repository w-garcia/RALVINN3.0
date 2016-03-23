# Q-Learning

This repository contains Matlab scripts for Q-Learning using q-tables as well as Python code for Q-Learning with neural networks. Known dependencies for the Python code include [Theano](http://deeplearning.net/software/theano/), [NumPy](http://www.numpy.org/), [Lasagne](https://github.com/Lasagne/Lasagne), and [Matplotlib](http://matplotlib.org/faq/usage_faq.html). To run the Python code, pull the repository, cd to the directory, and then run the test script:

```
$ python test.py
```

The script will output the cost function which should decrease with training iterations. At the very end, it tests whether it has learned the optimal path and prints out the path that the agent takes using a binary matrix. Finally, it visualizes the weights for the neural network that intuitively show what the agent has learned. 

