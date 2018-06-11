# Calibration methods 

import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time
from sklearn.metrics import log_loss
from keras.losses import categorical_crossentropy
from os.path import join
import sklearn.metrics as metrics

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.unpickle_probspy3 import unpickle_probs
from utility.evaluation import ECE, MCE

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    
    # Replace NaN with 0, as it should be close to zero  
    idx_nan = np.where(np.isnan(x))
    x[idx_nan] = 0

    # Add epsilon to zero x-values so there are no zeroes
    epsilon = 0.00000000000000000000000000000000000000001
    idx_zero = np.where(x == 0)
    x[idx_zero] = epsilon
    
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)
    
class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)
            

def evaluate(probs, y_true, verbose = False, normalize = False, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss), returns various scoring measures
    """
    
    # Replace NaN and zeroes with epsilon close to zero 
    idx_nan = np.where(np.isnan(probs))
    probs[idx_nan] = 0
    idx_zero = np.where(probs == 0)
    probs[idx_zero] = 0.00000000000000000000000000000000000000001
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    
    if normalize:
        confs = np.max(probs, axis=1)/(np.sum(probs, axis=1))

    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
    # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size = 1./bins)
    
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1./bins)
    
    loss = log_loss(y_true=y_true, y_pred=probs)
    
    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
    
    return (error, ece, mce, loss)
    
def cal_results(fn, path, files, m_kwargs = {}, approach = "all"):
    
    """
    Calibrate models scores, using output from logits files and given function (fn). 
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.
    
    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K approach.
        
    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    
    """
    
    df = pd.DataFrame(columns=["Name", "Error", "ECE", "MCE", "Loss"])
    
    total_t1 = time.time()
    
    for i, f in enumerate(files):
        
        name = "_".join(f.split("_")[1:-1])
        print(name)
        t1 = time.time()

        FILE_PATH = join(path, f)
        (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)
        
        if approach == "all":            

            y_val = y_val.flatten()

            model = fn(**m_kwargs)

            model.fit(logits_val, y_val)

            probs_test = model.predict(logits_test)
            
            # Replace NaN with epsilon close to zero, as it should be close to zero 
            idx_nan = np.where(np.isnan(probs_test))
            probs_test[idx_nan] = 0.00000000000000000000000000000000000000001
            
            error, ece, mce, loss = evaluate(softmax(logits_test), y_test, verbose=True)  # Test before scaling
            error2, ece2, mce2, loss2 = evaluate(probs_test, y_test, verbose=False)
            
        else:  # 1-vs-k models
            probs_val = softmax(logits_val)  # Softmax logits
            probs_test = softmax(logits_test)
            K = probs_test.shape[1]
            
            # Replace NaN with epsilon close to zero, as it should be close to zero 
            idx_nan = np.where(np.isnan(probs_test))
            probs_test[idx_nan] = 0.00000000000000000000000000000000000000001

            idx_nan = np.where(np.isnan(probs_val))
            probs_val[idx_nan] = 0.00000000000000000000000000000000000000001
            
            # Go through all the classes
            for k in range(K):
                # Prep class labels (1 fixed true class, 0 other classes)
                y_cal = np.array(y_val == k, dtype="int")[:, 0]

                # Train model
                model = fn(**m_kwargs)
                model.fit(probs_val[:, k], y_cal) # Get only one column with probs for given class "k"

                probs_test[:, k] = model.predict(probs_test[:, k])

            # Get results for test set
            error, ece, mce, loss = evaluate(softmax(logits_test), y_test, verbose=True, normalize=False)
            error2, ece2, mce2, loss2 = evaluate(probs_test, y_test, verbose=False, normalize=True)
                  
        df.loc[i*2] = [name, error, ece, mce, loss]
        df.loc[i*2+1] = [(name + "_calib"), error2, ece2, mce2, loss2]
        
        t2 = time.time()
        print("Time taken:", (t2-t1), "\n")
        
    total_t2 = time.time()
    print("Total time taken:", (total_t2-total_t1))
        
    return df
    