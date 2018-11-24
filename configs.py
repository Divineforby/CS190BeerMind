cfg = {}
cfg['input_dim'] = 216 # input dimension to LSTM
cfg['hidden_dim'] = 1024# hidden dimension for LSTM
cfg['output_dim'] = 111# output dimension of the model
cfg['layers'] = 2# number of layers of LSTM
cfg['dropout'] = 0# dropout rate between two layers of LSTM; useful only when layers > 1; between 0 and 1
cfg['bidirectional'] = False# True or False; True means using a bidirectional LSTM
cfg['batch_size'] = 0# batch size of input
cfg['learning_rate'] = 0.001# learning rate to be used
cfg['L2_penalty'] = 0# weighting constant for L2 regularization term; this is a parameter when you define optimizer
cfg['gen_temp'] = 1# temperature to use while generating reviews
cfg['max_len'] = 0# maximum character length of the generated reviews
cfg['epochs'] = 10# number of epochs for which the model is trained
cfg['cuda'] = True#True or False depending whether you want to run your model on a GPU or not. If you set this to True, make sure to start a GPU pod on ieng6 server
cfg['train'] = True# True or False; True denotes that the model is bein deployed in training mode, False means the model is not being used to generate reviews