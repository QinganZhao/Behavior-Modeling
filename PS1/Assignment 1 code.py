
# coding: utf-8

# In[11]:


# PS1 - CE264
# Sample Code
# GSIs: Mustapha Harb - Mengqiao Yu - Andrew Campbell

# importing the requried libraries
from collections import OrderedDict    # For recording the model specification 

import pandas as pd                    # For file input/output
import numpy as np                     # For vectorized math operations

import pylogit as pl                   # For MNL model estimation and
                                       # conversion from wide to long format


# reading the data file 
data_01 = pd.read_csv("data01.csv",sep=",")

# Look at the columns of the data file
data_01.columns

#look at the first 20 columns of the code
data_01.head(20)



# specifying the utility equations

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["intercept"] = [1, 2]
basic_names["intercept"] = ['ASC Train',
                            'ASC Metro']

basic_specification["travel_time_hrs"] = [[1, 2,], 3]
basic_names["travel_time_hrs"] = ['Travel Time, units:hrs (Train and Metro)',
                                  'Travel Time, units:hrs (Car)']
                                  
basic_specification["travel_cost_hundreth"] = [[1, 2,], 3]
basic_names["travel_cost_hundreth"] = ['Travel Cost, units:hundredth (Train and Metro)',
                                  'Travel Cost, units:hundredth (Car)']

basic_specification["headway_hrs"] = [1, 2]
basic_names["headway_hrs"] = ["Headway, units:hrs, (Train)",
                              "Headway, units:hrs, (Metro)"]



##########
# Determine the columns for: alternative ids, the observation ids and the choice
##########
# The 'alternative_id' variable will identify the alternative associated with each row.
alternative_id = "alt_id"

# The 'obs_id' variable will identify the observation id associated with each row.
observation_id = "obs_id"


# Create a 'choice' variable which identifies the choice associated with each row.
choice = "CHOICE"

# Estimate the multinomial logit model (MNL)
model_01_mnl = pl.create_choice_model(data=data_01,
                                        alt_id_col=alternative_id,
                                        obs_id_col=observation_id,
                                        choice_col=choice,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

# Specify the initial values and method for the optimization.
model_01_mnl.fit_mle(np.zeros(8)) # 8 is the total number of parameters to be esimtated



# Look at the estimation results
model_01_mnl.get_statsmodels_summary()

