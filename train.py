"""train.py
Usage:
    train.py <f_model_config> <f_opt_config>  [--prefix <p>] [--ce] [--db]
    train.py -r <exp_name> <idx> [--test]

Arguments:
 
Example:
    run train.py model/config/fc.yaml opt/config/sgd-128-lr.yaml --ce

Options:
"""
from __future__ import division
from docopt import docopt
import yaml
import torch
import tensorflow as tf
from torch import optim
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import os
import numpy as np

from utils import MiniBatcher#, MiniBatcherPerClass
import torchvision.models as tvm
import datetime
from opt.loss import *
from opt.nsgd import NoisedSGD
from model.fc import fc, lr
# from model.cnn import *
import cPickle as pkl
from cleverhans.utils_mnist import data_mnist
#### magic
NUM_VALID =10000 
tft = lambda x:torch.FloatTensor(x)
tfv = lambda x:Variable(tft(x))

def main(arguments):
    if arguments['-r']:
        exp_name = arguments['<exp_name>']
        f_model_config = 'model/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[0]+'.yaml'
        f_opt_config = 'opt/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[1]+'.yaml'
        old_exp_name = exp_name
        exp_name += '_resumed'
    else:
        f_model_config = arguments['<f_model_config>']
        f_opt_config = arguments['<f_opt_config>']
        model_name = os.path.basename(f_model_config).split('.')[0]
        opt_name = os.path.basename(f_opt_config).split('.')[0]
        timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
        data_name = 'mnist'
        if arguments['--prefix']:
            exp_name = '%s:%s-X-%s-X-%s@%s' % (arguments['<p>'], model_name, opt_name, data_name, timestamp)
        else:
            exp_name = '%s-X-%s-X-%s@%s' % (model_name, opt_name, data_name, timestamp)
        if arguments['--ce']:
            exp_name = 'CE.' + exp_name
        
    
    model_config = yaml.load(open(f_model_config, 'rb'))
    opt_config = yaml.load(open(f_opt_config, 'rb'))

    print ('\n\n\n\n>>>>>>>>> [Experiment Name]')
    print (exp_name)
    print ('<<<<<<<<<\n\n\n\n')

    ## Experiment stuff
    if not os.path.exists('./saves/%s'%exp_name):
        os.makedirs('./saves/%s'%exp_name)

    ## Data
    X, Y, X_test, Y_test = data_mnist() #(N, W, H, C) ...  
    Y_val = Y[-NUM_VALID:]
    X_val = X[-NUM_VALID:]
    Y = Y[:NUM_VALID]
    X = X[:NUM_VALID]

    X = tfv(X)
    Y = tft(Y)
    # Dataset (X size(N,D) , Y size(N,K))
    ## Model
    model = eval(model_config['name'])(**model_config['kwargs'])
    model.type(torch.cuda.FloatTensor)
    ## Optimizer
    opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])


    if arguments['-r']:
        model.load('./saves/%s/model_%s.t7'%(old_exp_name,arguments['<idx>']))
        opt.load_state_dict(torch.load('./saves/%s/opt_%s.t7'%(old_exp_name,arguments['<idx>'])))

        if arguments['--test']:
            raise NotImplementedError()


    ## tensorboard
    #ph
    ph_accuracy = tf.placeholder(tf.float32,  name='accuracy')
    ph_loss = tf.placeholder(tf.float32,  name='loss')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    tf_acc = tf.summary.scalar('accuracy', ph_accuracy)
    tf_loss = tf.summary.scalar('loss', ph_loss)
    
    log_folder = os.path.join('./logs', exp_name)
    # remove existing log folder for the same model.
    if os.path.exists(log_folder):
        import shutil
        shutil.rmtree(log_folder, ignore_errors=True)

    sess = tf.InteractiveSession()   

    train_writer = tf.summary.FileWriter(os.path.join(log_folder, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)

    batcher = eval(opt_config['batcher_name'])(X.size()[0], **opt_config['batcher_kwargs'])

    ## Loss
    if arguments['--ce']:
        Loss = CE()
    else:
        raise NotImplementedError()
    

    best_val_acc = 0
    val_errors = []
    tf.global_variables_initializer().run()
    if not arguments['--db']:
        ## Algorithm
        for idx in tqdm(xrange(opt_config['max_train_iters'])):
            if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == idx:
                _, tmp_fac = opt_config['lrsche'].pop(0)
                sd = opt.state_dict()
                assert len(sd['param_groups']) ==1
                sd['param_groups'][0]['lr'] *= tmp_fac
                opt.load_state_dict(sd)

                
            idxs = batcher.next(idx)
            X_batch = X[torch.LongTensor(idxs)].type(torch.cuda.FloatTensor)
            Y_batch = Y[torch.LongTensor(idxs)]#.type(torch.cuda.FloatTensor)
            ## network
            tv_F = model.forward(X_batch)
            F = tv_F.data.clone().type(torch.FloatTensor)
            ### loss layer
            loss, G, train_pred = Loss.train(F, Y_batch)

            model.zero_grad()
            tv_F.backward(gradient=G.type(torch.cuda.FloatTensor))
            opt.step()


            # TensorBoard
            #accuracy
            train_gt = Y[torch.LongTensor(idxs)].numpy().argmax(1)
            train_accuracy = (train_pred[batcher.start_unlabelled:] == train_gt[batcher.start_unlabelled:]).mean()

            # summarize
            acc= sess.run(tf_acc, feed_dict={ph_accuracy:train_accuracy})
            loss = sess.run(tf_loss, feed_dict={ph_loss:loss})
            tmp = Y_batch.numpy()
            train_writer.add_summary(acc+loss, idx)

            #validate
            if idx>0 and idx%200==0:
                def _validate_batch(model, X_val_batch, Y_val_batch):
                    model.eval()
                    val_pred = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.cuda.FloatTensor))
                    val_accuracy = np.mean(Y_val_batch.argmax(1) == val_pred)
                    model.train()
                    return val_accuracy
                
                val_batch_size = batcher.batch_size
                val_batches = Y_val.shape[0] // val_batch_size
                v1 = []
                for vidx in xrange(val_batches):
                    val_accuracy = _validate_batch(model, X_val[vidx*val_batch_size:(vidx+1)*val_batch_size], Y_val[vidx*val_batch_size:(vidx+1)*val_batch_size])
                    v1.append(val_accuracy)
                val_accuracy = np.mean(v1)
                print (val_accuracy)
                acc= sess.run(tf_acc, feed_dict={ph_accuracy:val_accuracy})
                val_writer.add_summary(acc, idx)
                val_errors.append(val_accuracy)
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    name = './saves/%s/model_best.t7'%(exp_name)
                    print ("[Saving to]")
                    print (name)
                    model.save(name)
            ## checkpoint
            if idx>0 and idx%1000==0:
                name = './saves/%s/model_%i.t7'%(exp_name,idx)
                print ("[Saving to]")
                print (name)
                model.save(name)
                torch.save(opt.state_dict(), './saves/%s/opt_%i.t7'%(exp_name,idx))
    pkl.dump(val_errors, open(os.path.join(log_folder, 'val.log'), 'wb'))
    return best_val_acc


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    main(arguments)

