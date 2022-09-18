# SimpleAlgoTrade Model

The goal of this project to create crypto price prediction
for short interval time.

Market prices for short interval (< 1 day) usually contains
a lot of noise, thus the main scope of this project to implement and validate
a few ideas:

1) Test whether some Technical Analysis indicators contains
   information which can be picked by a model
2) The model quality is estimated using information coefficient,
   for simplicity the linear spearman correlation coefficient is used
3) The overall model quality is estimated by using vectorized back tester

Model:
1) In total 83 features were produced
2) The main candidate for the model was chosen to be Light GBM regressor,
   since it does not put any statistical assumption on the data as it is the
   case for classical Time-Series models and GLM, as well as GBM can be easily generalized
   to complex data structure
3) Since the data is extremely noise, the boosted trees may overfit and start
   modelling the noise, for this purpose certain specific parameters (subsample and reg-lambda etc...)
   were used in hyper-parameter tuning using bayesian optimisation
   

The Cross-Validation Result:

1) The model has high variance, meaning that the performance on in-sample data
is highly better than on out-of-sample data. There exist certain indication, that
bigger sample size might decrease variance
2) The model produces out-of-sample result which demonstrates a martingale 
   process such that E[X_{n+1}|F_n] = X_{n}
3) The model interpretation using partial dependency did not reveals strong 
  pattern by visual inspection. Note partial dependency does not take under
  consideration the correlation in between the features

The Back-Test Result:


What can be improved:
1) Fetch more Data
2) Create a logic for merging the last exercise price with full trading book information
3) Create more features from trading book data such as Volume, Bid-ask-Spread, VWAP on bid and ask etc...
4) Fetch event data

