```python
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import anderson
import statsmodels
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import math
```


```python
def CreateResidualAnalysis(residuals,fits, exog, labeladj=.05):
    '''
    This takes the output of an OLS fitted model to produce a graphical and statistical residual analysis.
    '''
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    plt.title('Residuals vs Fits')
    plt.xlabel('Fits')
    plt.ylabel('Residuals')
    
    color = []
    for i, txt in enumerate(data.index):
        if (residuals[i] > 2*residuals.std()) or (residuals[i] <   -2*residuals.std()):
            ax.annotate(txt, (fits[i], residuals[i]),xytext=(fits[i] +labeladj*fits.std() , residuals[i]+labeladj*residuals.std()))
            col = 'red'
        else: 
            col = 'blue'
        color.append(col)
        
    plt.scatter(fits,residuals, color=color)
    
    
    ad_results = anderson(residuals)
    
    if ad_results[0] > ad_results[1][3]:
        message = r'Using the Anderson-Darling on the residuals with $\alpha = .05$, '+ str(np.round(ad_results[0],3))+ ' > '+ str(ad_results[1][3])+'.\n Therefore the residuals are not normal.'
        plt.text(.5, -.25, message, horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)
    
    
    else:
        message = r'Using the Anderson-Darling on the residuals with $\alpha = .05$, '+ str(np.round(ad_results[0],3))+ ' < '+ str(ad_results[1][3])+'.\n Therefore the residuals are normal.'
        plt.text(.5, -.25, message, horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)


    names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']

    test =statsmodels.stats.diagnostic.het_breuschpagan(residuals,exog)
    
    
    if test[1] > 0.05:
        bp_message = 'The p-value of the Breusch-Pagan test is ' + str(test[1]) + '. \n Therefore, there is heteroskedasticity within the residuals.'
        plt.text(.5, -.4, bp_message, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:
        bp_message = 'The p-value of the Breusch-Pagan test is ' + str(test[1]) + '. \n Therefore, there is homoskedacity within the residuals.'
        plt.text(.5, -.4, bp_message, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    plt.show()

```


```python
def OneOLSRun(x,y):
    '''
    We want to take in a set of columns in the X. And a column in the y. 
    The output should be a one off output of the residual analysis and linear regression results.
    '''
    x = sm.add_constant(x)
    model = sm.OLS(y.astype(float),x.astype(float))
    results = model.fit()
    print(results.summary(alpha=.05))
    CreateResidualAnalysis(results.resid,results.fittedvalues,results.model.exog)
```


```python
def Scale(df,continuous,categorical):
    '''
    Take in columns of continuous X's and standardize them.
    Reurn standardaized df.
    '''
    new_df = pd.DataFrame()
    for name in continuous:
        cur_col = df[name]
        cur_std = cur_col.std()
        cur_mean = cur_col.mean()
        cur_col_standardized = (cur_col - cur_mean)/cur_std
        new_df[str(name+' Standardized')] = cur_col_standardized
    for name in categorical:
        new_df[name]= df[name]
        
    return new_df
```


```python
def CreateInteractions(df):
    '''
    Input a dataframe
    Return a dataframe with each column multiplied by all other columns in the dataframe
    
    '''
    temp_df = df.copy()
    cols = temp_df.columns
    for i in range(len(cols)-1):
        cur_col = cols[i]
        next_cols = cols[i+1:]
        for j in range(len(next_cols)):
            interaction = cur_col+'*'+next_cols[j]
            temp_df[interaction] =  temp_df[cur_col]*temp_df[next_cols[j]] #pay CLOSE attention to operation
    return temp_df
```


```python
def FSW(df, x_totest,y,alpha,x=None):
    '''x_totest are column names to test. X is set to none, but represents the name of X's to force into the model.'''
    '''return columns names to be used?'''
    # print(x)
    alphas = []
    current_testing = []
    if x is None: #if we are starting out with a single term model        
        for column_index in range(len(x_totest)):
            temp_x = df[x_totest[column_index]]
            temp_x = sm.add_constant(temp_x)
            # print('x is ', x_totest[column_index])
            model = sm.OLS(y.astype(float),temp_x.astype(float)) #did as float
            results = model.fit()
            current_p_value = results.pvalues.iloc[-1]
            if math.isnan(current_p_value):
                current_p_value = 1000
            alphas.append(current_p_value)
        best_alpha_index = np.argmin(alphas)
        # print(best_alpha_index)
        # print(alphas)
        if alphas[best_alpha_index] < .05:
            x = [x_totest[best_alpha_index]] #add to required
            x_totest.remove(x_totest[best_alpha_index]) #remove from testing
            return FSW(df, x_totest,y,alpha,x)
        else:
            return 'No terms meet the alpha threshold for a single term model'
    else: #case when we have at least a single order model/terms that must be in the new model.
        initial_df = df[x] #create a subset of the x's that have already met the threshold.
        for column_index in range(len(x_totest)):
            #temp_x = df[x_totest[column_index]]
            # print(x_totest[column_index])
            combined_df = initial_df.copy()
            combined_df[x_totest[column_index]] = df[x_totest[column_index]]
            combined_df = sm.add_constant(combined_df)
            model = sm.OLS(y.astype(float),combined_df.astype(float))
            results = model.fit()
            current_p_value = results.pvalues.iloc[-1]
            if math.isnan(current_p_value):
                current_p_value = 1000
            alphas.append(current_p_value)
        best_alpha_index = np.argmin(alphas)
        # print(best_alpha_index)
        # print(alphas)
        if alphas[best_alpha_index] < .05:
            # print(best_alpha_index)
            # print(alphas)
            x.append(x_totest[best_alpha_index]) #add to required
            x_totest.remove(x_totest[best_alpha_index]) #remove from testing
            return FSW(df, x_totest,y,alpha,x) 
        else:
            return x

```


```python

```
