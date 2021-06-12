Classification with Keras
https://www.pluralsight.com/guides/classification-keras

Code examples
https://keras.io/examples/

tensorflow code sample
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#applying_this_tutorial_to_your_problem
包含数据预处理

Introduction to Keras, Part One: Data Loading
https://towardsdatascience.com/introduction-to-keras-part-one-data-loading-43b9c015e27c


A detailed example of how to use data generators with Keras
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly



Residual Networks (ResNet) – Deep Learning  
https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/?ref=rp
mark:python code sample for residential network


good explan for transformer
https://www.youtube.com/watch?v=TQQlZhbC5ps


理解语言的 Transformer 模型
https://www.tensorflow.org/tutorials/text/transformer



Timeseries classification from scratch
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Compare the effect of different scalers on data with outliers
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py


Time series forecasting
https://www.tensorflow.org/tutorials/structured_data/time_series


How to Scale Data for Long Short-Term Memory Networks in Python
https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/

How to Use Convolutional Neural Networks for Time Series Classification
https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57


The answer to your question is: yes, efficacy of a model depends on scaling. It's very important to scale your variables in the right range and combine them with the right activation function.
The reason is the following: the power of Neural Networks is due to the fact that they can learn any non-linear regularity of your data. This depends on the use of non-linear activation functions (tanh, ReLU, ELU, you name it). However, most of activation functions tend to behave in a non-linear way only around zero. Take the plot of a ReLU, for example. If you move further away from zero (in both directions) the function becomes very "linear" (i.e. its derivative is a constant).
All common activation functions tend to behave like this: non-linear (i.e. very powerful) in the locality of zero, and very linear (or flat) further away from zero. That is why all data are usually scaled in the [0, 1] or in the [-1, 1] range. In this way, activation functions can give their best, and Neural Networks can learn all the most complex patterns in your data.
When you work with CNNs, for example, most of pixel data come in the [0, 255] range. This is very bad for all activation functions, since between 0 and 255 pretty much any of them will look almost completely linear. In this way, your CNN wouldn't be able to learn much.
https://datascience.stackexchange.com/questions/40535/efficacy-of-model-depends-on-scaling/40536