# Automating an Ordinary Least Squares Analysis

## Project Description
    
As a part of my graduate course work, I took the class 'Regression' which takes a deep dive into the theoretical frame work of ordinary least squares, a foundation for what many people think of as a linear regression. The final project of this class was to perform a regression analysis on data found externally. I decided to focus on analyzing factors that influence the total contract amount of the active roster within the Commanders from the starting season of 2023. 

For the class, I performed the project mainly in Minitab, a commericial statistical software. However, I wanted to translate the results of the project into Python, demonstrating how to perform OLS analysis through automated Python code. 

Without going into the weeds too much in the analysis and just copy over my whole project, I will focus on highlights within the work, and at the end, I will describe the functionality of my different functions. For the python portion, I created functions that scales your continuous features, creates interactions between variables, performs one off OLS regression with tests for residual normality and homoskedacisty, and a forward stepwise regression. 

It is beneficial to have this sort of work automated in Python because linear regressions allow for easily explainable variable impacts. There is a less of a black box since it is easier to see that as one variable increases, the other decreases. Though a linear regression isn't for every problem, with proper attention to details, it can be adapted to many of them.

The code for those functions can be located [here](RegressionFunctions.md)

## Looking at the data
I obtained my data from spotrac.com which contains copius amounts of sports related data.
- URL: https://www.spotrac.com/nfl/washington-commanders/contracts/

For this dataset, there was not a simple download capability, so I performed the first part of my data cleaning within Excel. Here I created categorical variables for whether a player was a quarter back, on the defense or offense (or special teams), and whether or not they were drafted. Likewise, some players were not drafted, so if they were not drafted, I placed their draft round as 8, which doesn't exist but follows the draft round logic that they came "last."  I did a similar adjustment for draft pick.


```python
#import regression functions
%run RegressionFunctions.ipynb
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
#read in data
data= pd.read_excel('./Data/CommandersSalaryInfo.xlsx',sheet_name='Sheet1')
display(data)
```


![png](UnAdjst_Df.png)


The dataframe above demonstrates that an Excel to Python pipepline is not always smooth. Especially in a business setting where data scientists are asked to look at cohorts of unrelated Excel spreadsheets, it is important to identify rows that need to be dropped.


```python
data.drop(['Unnamed: 7'],axis=1,inplace=True)
data = data.iloc[:67]
```


```python
display(data)
```


![png](Adjusted_df.png)




```python
sns.violinplot(data, y='Contract Amount',x='Drafted',cut=0,fill=False,inner='quart')
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xticks([0,1],['Not Drafted','Drafted'])
plt.title('Distribution of Total Contract Amount by Draft Status')
plt.show()
```


    
![png](output_8_0.png)
    



```python
sns.violinplot(data, y='Contract Amount',x='Offense',cut=0,fill=False,inner='quart')
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.xticks([0,1],['Not Offense','Offense'])
plt.title('Distribution of Total Contract Amount by Offensive Status')
plt.xlabel('')
plt.show()
```


    
![png](output_9_0.png)
    


The above plots are important ones. They tell us that there is a substantial difference in range within the total contract amounts of players that were drafted versus players that were not. Though there is a range difference between offensive and non-offensive players (special team players included), that difference appears to be less than the draft status difference.

After looking into the data, it is clear that the top payed players include Jonathen Allen, Darron Payne, and Terry Mclaurin. Specifically, the Commanders are paying certain defensive linemen and wide recievers more than other notable positions such as quarterbacks which usually seem like they would be the highest payed player. This makes sense given the Commanders' current starting quarterback is still on his rookie contract. 

Note, the lines within the plot demonstrate the 25th, 50th, and 75th percentiles. This indicates that the vast majority of players of total contract amounts of less than $20,000,000 and that there are magnitudle differences in the y's. Withouth going too fast, the magnitude differences in the contract amounts indicates to me that there may need to be a y-transformation. 


```python
sns.heatmap(data[['Contract Amount','Contract Years','Age','Draft Round Adjusted','Draft Pick']].corr(),cmap='flare',annot=True)
plt.title('Correlation Matrix of Continuous Variables')
plt.show()
```


    
![png](output_11_0.png)
    


A problem with OLS is that multi-collinearity within the data can negatively impact the model. Specifically, multi-collinearity, can increase the standard error of the beta's which in turn increases the range of values that a beta can take on with a specified alpha level. 

Also, looking at the original dataframe, our x's have different scales. A method I use to adjust for this is to standardize the data by subtracting the mean and dividing by the standard deviation for continuous predictors.

## Model Building

Generally, to start model building, I like to run all my variables through a single order regression and see what happens. 


```python
X = data[['Age', 'Contract Years', 'Draft Round Adjusted', 'Drafted',
       'Years of Experience', 'Draft Pick','Offense','Defense ','QB']]
y= data['Contract Amount']
OneOLSRun(X,y)
```

    /opt/conda/lib/python3.11/site-packages/statsmodels/regression/linear_model.py:1965: RuntimeWarning: divide by zero encountered in scalar divide
      return np.sqrt(eigvals[0]/eigvals[-1])


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        Contract Amount   R-squared:                       0.446
    Model:                            OLS   Adj. R-squared:                  0.370
    Method:                 Least Squares   F-statistic:                     5.848
    Date:                Fri, 05 Jan 2024   Prob (F-statistic):           1.78e-05
    Time:                        18:19:25   Log-Likelihood:                -1190.1
    No. Observations:                  67   AIC:                             2398.
    Df Residuals:                      58   BIC:                             2418.
    Df Model:                           8                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    const                -2.261e+06   4.16e+07     -0.054      0.957   -8.56e+07    8.11e+07
    Age                   3.581e+05   1.56e+06      0.230      0.819   -2.76e+06    3.48e+06
    Contract Years        5.518e+06   1.59e+06      3.470      0.001    2.33e+06     8.7e+06
    Draft Round Adjusted -9.625e+06   6.12e+06     -1.572      0.121   -2.19e+07    2.63e+06
    Drafted              -1.367e+07   6.79e+06     -2.014      0.049   -2.73e+07   -8.65e+04
    Years of Experience   2.961e+06   1.43e+06      2.065      0.043    9.01e+04    5.83e+06
    Draft Pick            1.635e+05   1.62e+05      1.011      0.316    -1.6e+05    4.87e+05
    Offense               1.014e+07   8.93e+06      1.136      0.260   -7.72e+06     2.8e+07
    Defense                1.49e+07   9.21e+06      1.617      0.111   -3.54e+06    3.33e+07
    QB                            0          0        nan        nan           0           0
    ==============================================================================
    Omnibus:                       49.644   Durbin-Watson:                   0.675
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              187.832
    Skew:                           2.246   Prob(JB):                     1.63e-41
    Kurtosis:                       9.863   Cond. No.                          inf
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is      0. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.



    
![png](output_14_2.png)
    


### Inital Model Breakdown + Theory

For a linear regression, the residuals (difference between the fitted value and the true value), must have a normal distribution, have a constant variance, and cannot be linearly dependent with the first two being very critical. 

I created a function OneOLSRun, that is primarily run on top of statsmodels' api for OLS. I rely on that library in order to access important values such as the R-Squared and R-Squared Adjusted which explain the amount of variation that the model accounts for and the amount of variation the model accounts for per degree of freedom (i.e. how much more each independent variable accounts for given other variables are in the model), respectively. Likewise, this library allows for diagnosics such as the Durbin-Watson statistic, conditional number, AIC/BIC, and the marginal t-test for our beta's.

So on top of displaying the model summary, I added components within my function that displays a plot of the residuals vs. the fits. This plot allows us to see if there is a constant variance. If a residual was farther than 2 standard deviations from the mean residual, I colored it red and marked it with its index for further inspect, as that is an outlying residual. 

Likewise, I output the results of the Anderson-Darling tests, a test for normality, and a Breusch-Pagan test, a test for homoskedacity, within the residuals. 

This output tells us a few things. There is super high multi-collinearity (look at the conditional number). The residuals are non-normal, and there is non-constant variance. The model only accounts for 44% of the variation within the contract amount. Likewise, a QB shouldn't be a predictor because there aren't enough QB points.

## Data Transformations and Scaling

To adjust for the heteroskedacity, I took the natural log of the contract amount. I also scaled the x's in order to get them in the same scale.


```python
X = data[['Age', 'Contract Years', 'Draft Round Adjusted', 'Drafted',
       'Years of Experience', 'Draft Pick','Offense','Defense ','QB']]
y= data['Contract Amount']
Scaled_X = Scale(data,['Age', 'Contract Years', 'Draft Round Adjusted', 'Years of Experience','Draft Pick'],
                ['Drafted','Offense','Defense ', 'QB'])
display(Scaled_X.head(2))
```


![png](Scaled.png)


We might also be missing information from interactions, so I placed interactions within the dataframe.


```python
Int_Scaled_X = CreateInteractions(Scaled_X)
Int_Scaled_X.columns
```




    Index(['Age Standardized', 'Contract Years Standardized',
           'Draft Round Adjusted Standardized', 'Years of Experience Standardized',
           'Draft Pick Standardized', 'Drafted', 'Offense', 'Defense ', 'QB',
           'Age Standardized*Contract Years Standardized',
           'Age Standardized*Draft Round Adjusted Standardized',
           'Age Standardized*Years of Experience Standardized',
           'Age Standardized*Draft Pick Standardized', 'Age Standardized*Drafted',
           'Age Standardized*Offense', 'Age Standardized*Defense ',
           'Age Standardized*QB',
           'Contract Years Standardized*Draft Round Adjusted Standardized',
           'Contract Years Standardized*Years of Experience Standardized',
           'Contract Years Standardized*Draft Pick Standardized',
           'Contract Years Standardized*Drafted',
           'Contract Years Standardized*Offense',
           'Contract Years Standardized*Defense ',
           'Contract Years Standardized*QB',
           'Draft Round Adjusted Standardized*Years of Experience Standardized',
           'Draft Round Adjusted Standardized*Draft Pick Standardized',
           'Draft Round Adjusted Standardized*Drafted',
           'Draft Round Adjusted Standardized*Offense',
           'Draft Round Adjusted Standardized*Defense ',
           'Draft Round Adjusted Standardized*QB',
           'Years of Experience Standardized*Draft Pick Standardized',
           'Years of Experience Standardized*Drafted',
           'Years of Experience Standardized*Offense',
           'Years of Experience Standardized*Defense ',
           'Years of Experience Standardized*QB',
           'Draft Pick Standardized*Drafted', 'Draft Pick Standardized*Offense',
           'Draft Pick Standardized*Defense ', 'Draft Pick Standardized*QB',
           'Drafted*Offense', 'Drafted*Defense ', 'Drafted*QB', 'Offense*Defense ',
           'Offense*QB', 'Defense *QB'],
          dtype='object')




```python
y = np.log(data['Contract Amount'])
```


```python
OneOLSRun(Int_Scaled_X,y)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        Contract Amount   R-squared:                       0.864
    Model:                            OLS   Adj. R-squared:                  0.751
    Method:                 Least Squares   F-statistic:                     7.635
    Date:                Fri, 05 Jan 2024   Prob (F-statistic):           1.71e-08
    Time:                        18:19:26   Log-Likelihood:                -37.802
    No. Observations:                  67   AIC:                             137.6
    Df Residuals:                      36   BIC:                             206.0
    Df Model:                          30                                         
    Covariance Type:            nonrobust                                         
    ======================================================================================================================================
                                                                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------------------------------------------------
    const                                                                 11.1275      0.651     17.089      0.000       9.807      12.448
    Age Standardized                                                       0.2309      0.660      0.350      0.728      -1.107       1.568
    Contract Years Standardized                                            0.3383      0.632      0.535      0.596      -0.944       1.621
    Draft Round Adjusted Standardized                                      4.7450      0.639      7.428      0.000       3.450       6.040
    Years of Experience Standardized                                      -0.1977      0.525     -0.377      0.709      -1.263       0.867
    Draft Pick Standardized                                               -1.7401      1.403     -1.240      0.223      -4.585       1.105
    Drafted                                                                1.3121      0.978      1.341      0.188      -0.672       3.296
    Offense                                                                2.8745      0.659      4.359      0.000       1.537       4.212
    Defense                                                                3.1233      0.509      6.136      0.000       2.091       4.156
    QB                                                                  1.323e-14   4.93e-15      2.684      0.011    3.23e-15    2.32e-14
    Age Standardized*Contract Years Standardized                           0.5944      0.307      1.934      0.061      -0.029       1.218
    Age Standardized*Draft Round Adjusted Standardized                     0.2990      2.193      0.136      0.892      -4.149       4.747
    Age Standardized*Years of Experience Standardized                      0.1668      0.139      1.198      0.239      -0.116       0.449
    Age Standardized*Draft Pick Standardized                              -0.2365      2.022     -0.117      0.908      -4.337       3.864
    Age Standardized*Drafted                                              -1.0833      1.061     -1.021      0.314      -3.235       1.068
    Age Standardized*Offense                                               0.9410      0.426      2.209      0.034       0.077       1.805
    Age Standardized*Defense                                               0.6656      0.463      1.436      0.160      -0.274       1.605
    Age Standardized*QB                                                -1.148e-14   3.48e-15     -3.300      0.002   -1.85e-14   -4.43e-15
    Contract Years Standardized*Draft Round Adjusted Standardized          0.7579      1.159      0.654      0.517      -1.593       3.109
    Contract Years Standardized*Years of Experience Standardized          -0.2640      0.265     -0.996      0.326      -0.801       0.273
    Contract Years Standardized*Draft Pick Standardized                   -0.8529      1.072     -0.795      0.432      -3.027       1.322
    Contract Years Standardized*Drafted                                    0.0271      0.560      0.048      0.962      -1.108       1.163
    Contract Years Standardized*Offense                                    0.4197      0.436      0.963      0.342      -0.464       1.304
    Contract Years Standardized*Defense                                    0.4635      0.485      0.957      0.345      -0.519       1.446
    Contract Years Standardized*QB                                      1.532e-16    5.4e-16      0.283      0.779   -9.43e-16    1.25e-15
    Draft Round Adjusted Standardized*Years of Experience Standardized    -1.7127      1.782     -0.961      0.343      -5.326       1.900
    Draft Round Adjusted Standardized*Draft Pick Standardized              0.2619      0.184      1.423      0.163      -0.111       0.635
    Draft Round Adjusted Standardized*Drafted                             -6.7662      1.421     -4.762      0.000      -9.648      -3.885
    Draft Round Adjusted Standardized*Offense                              0.8054      0.839      0.960      0.343      -0.895       2.506
    Draft Round Adjusted Standardized*Defense                             -0.0391      0.852     -0.046      0.964      -1.768       1.689
    Draft Round Adjusted Standardized*QB                                 3.99e-16   2.41e-16      1.653      0.107   -9.05e-17    8.89e-16
    Years of Experience Standardized*Draft Pick Standardized               1.4121      1.660      0.851      0.400      -1.954       4.778
    Years of Experience Standardized*Drafted                               0.5982      0.948      0.631      0.532      -1.323       2.520
    Years of Experience Standardized*Offense                              -0.0556      0.437     -0.127      0.899      -0.941       0.830
    Years of Experience Standardized*Defense                               0.2102      0.442      0.476      0.637      -0.686       1.106
    Years of Experience Standardized*QB                                         0          0        nan        nan           0           0
    Draft Pick Standardized*Drafted                                        6.2274      2.557      2.435      0.020       1.041      11.414
    Draft Pick Standardized*Offense                                       -3.6812      0.974     -3.778      0.001      -5.658      -1.705
    Draft Pick Standardized*Defense                                       -2.8590      1.063     -2.689      0.011      -5.015      -0.703
    Draft Pick Standardized*QB                                                  0          0        nan        nan           0           0
    Drafted*Offense                                                       -0.3037      0.547     -0.555      0.582      -1.414       0.807
    Drafted*Defense                                                       -0.8342      0.488     -1.709      0.096      -1.824       0.156
    Drafted*QB                                                                  0          0        nan        nan           0           0
    Offense*Defense                                                             0          0        nan        nan           0           0
    Offense*QB                                                                  0          0        nan        nan           0           0
    Defense *QB                                                                 0          0        nan        nan           0           0
    ==============================================================================
    Omnibus:                       39.446   Durbin-Watson:                   1.876
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              157.509
    Skew:                           1.629   Prob(JB):                     6.27e-35
    Kurtosis:                       9.768   Cond. No.                     1.34e+16
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.5e-30. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.



    
![png](output_22_1.png)
    


This didn't perform as well either. Even though we have a higher R-Squared, that is to come with adding more variables, so we want to be selective of how we add variables. Likewise, our high multi-collinearity is likely also affecting our model. 

The heteriskedacity, though, appears to have come down a bit but is still an issue.

## Feature selection - Forward Stepwise Regression

For feature selection there is a number of methods to employ. If you are dealing with multi-collinearity, a ridge regression can be used. That can overcomplicate the model though, as it might be more difficult to explain how a ridge regression decreases multi-collinearity and that the beta's have a slightly different meaning.

Because of this, I use a forward stepwise regression (FSR). An FSR starts by building a single term model for each variable. Then, it looks at the variable with the smallest p-value and compares it to a preset alpha value. Alpha is the likelihood of rejecting the null when the null is true (i.e. saying that the beta is not 0 when it actually is). If the p-value < alpha, then that term is added to the model. Then for the second round, the FSR method adds every other variable to the model, with the initial variable included. Then looking at the marginal t-test of the added variable, it performs the same task as above. By the end, some terms may no longer be signficiant, but I can drop those terms manually.

It is important to note that there a stepwise regression allows to drop variables, a backward stepwise regression goes in the reverse direction, and measures can be made to ensure hierarchy. I decided to only include the end results of my project and manually add hierarchy when needed.

I built my own FSW function because I found that current libraries do not have much capabilities to discriminate based on notable statistical methods. Likewise, I found certain libraries that did employ statistical methods (like comparing p-values), but I did not see the results produced via Minitab, which I am taking to be the check of my work. On top of that, many of the FSW functions built relied on While-loops. I created a FSW function recursively, which I though was the most efficient way for a FSW regression. 


```python
FSW(Int_Scaled_X,Int_Scaled_X.columns.to_list(),y,.05)
```




    ['Contract Years Standardized',
     'Years of Experience Standardized',
     'Draft Round Adjusted Standardized*Drafted',
     'Age Standardized*Contract Years Standardized']




```python
OneOLSRun(Int_Scaled_X[['Contract Years Standardized',
 'Years of Experience Standardized',
 'Draft Round Adjusted Standardized*Drafted',
 'Age Standardized*Contract Years Standardized']],y)

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        Contract Amount   R-squared:                       0.742
    Model:                            OLS   Adj. R-squared:                  0.725
    Method:                 Least Squares   F-statistic:                     44.47
    Date:                Fri, 05 Jan 2024   Prob (F-statistic):           1.46e-17
    Time:                        18:19:27   Log-Likelihood:                -59.352
    No. Observations:                  67   AIC:                             128.7
    Df Residuals:                      62   BIC:                             139.7
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ================================================================================================================
                                                       coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------------------------
    const                                           15.1962      0.089    171.542      0.000      15.019      15.373
    Contract Years Standardized                      0.7915      0.079     10.049      0.000       0.634       0.949
    Years of Experience Standardized                 0.6196      0.078      7.910      0.000       0.463       0.776
    Draft Round Adjusted Standardized*Drafted       -0.5464      0.105     -5.212      0.000      -0.756      -0.337
    Age Standardized*Contract Years Standardized     0.2116      0.087      2.425      0.018       0.037       0.386
    ==============================================================================
    Omnibus:                        4.861   Durbin-Watson:                   1.312
    Prob(Omnibus):                  0.088   Jarque-Bera (JB):                4.222
    Skew:                           0.418   Prob(JB):                        0.121
    Kurtosis:                       3.902   Cond. No.                         1.93
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    
![png](output_26_1.png)
    


Here we finally have normal residuals, homoskedacity, and low multi-collinearity. Through Testing in Minitab, I finalized the following model.

## Final Model


```python
OneOLSRun(Int_Scaled_X[['Contract Years Standardized',
 'Years of Experience Standardized',
 'Draft Round Adjusted Standardized', 'Drafted',
 'Age Standardized*Contract Years Standardized']],y)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        Contract Amount   R-squared:                       0.745
    Model:                            OLS   Adj. R-squared:                  0.724
    Method:                 Least Squares   F-statistic:                     35.71
    Date:                Fri, 05 Jan 2024   Prob (F-statistic):           6.70e-17
    Time:                        18:19:27   Log-Likelihood:                -58.856
    No. Observations:                  67   AIC:                             129.7
    Df Residuals:                      61   BIC:                             142.9
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ================================================================================================================
                                                       coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------------------------
    const                                           15.9895      0.204     78.536      0.000      15.582      16.397
    Contract Years Standardized                      0.8193      0.084      9.748      0.000       0.651       0.987
    Years of Experience Standardized                 0.6107      0.079      7.735      0.000       0.453       0.769
    Draft Round Adjusted Standardized               -0.5726      0.108     -5.279      0.000      -0.790      -0.356
    Drafted                                         -0.8528      0.254     -3.355      0.001      -1.361      -0.345
    Age Standardized*Contract Years Standardized     0.2058      0.088      2.351      0.022       0.031       0.381
    ==============================================================================
    Omnibus:                        6.041   Durbin-Watson:                   1.383
    Prob(Omnibus):                  0.049   Jarque-Bera (JB):                5.964
    Skew:                           0.445   Prob(JB):                       0.0507
    Kurtosis:                       4.159   Cond. No.                         6.20
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    
![png](output_29_1.png)
    


### Justification
It is clear that my final model is not necessarily my best model when fit to training data. I say this because my final model has a slightly higher AIC/BIC and conditional number. However, I emphasize the slight increase AIC/BIC. My goal is to have an overall simplistic model, and the interaction between draft status and draft round adds complication. Likewise, not including their first order terms goes against heirachy, but including them, because one is a categorical variable, does not work with the interaction, so I opted to rely on the first order terms. 

The increased conditional number makes sense since a drafted player will have a lower draft round pick than an undrafted player which I have designed to have the highest round. I did not apply hierachy to age because that increased the conditional number too much. 

Though not included in this display, I tested my model with three former Commander's players with different contract lengths and 2 of 3 terms fell within the 95% prediction interval. The one that did not fall in was the Kirk Cousin's contract extension which can be considered to depict the unusual nature of the contract.

### Conclusion
The biggest practical takeaway is that if a player wants a higher contract amount, they should commit to working more years. If they were not drafted, they can generally expect lower contract amounts, similar to those who were drafted later on. Otherwise, it is clear that years of experience and age generally have a positive impact on the contract amount. 

Lastly, the point of this notebook was to simply demonstrate the capability of python. Many linear regression packages do not display desired statistical outputs. Those that do, are not consolidated. Likewise, there are limited feature selection techniques that work exactly the way you want. So I demonstrated that it is feasible to get such results by combining multiple packages and techniques together.

This work leveraged sources such as GeekforGeeks and StackOverFlow as well as my education during my Master's degree. Relying on multiple sources is critical for employment functions. 


```python

```
