
# coding: utf-8

# # Examples and Exercises from Think Stats, 2nd Edition
# 
# http://thinkstats2.com
# 
# Copyright 2016 Allen B. Downey
# 
# MIT License: https://opensource.org/licenses/MIT
# 

# In[1]:


from __future__ import print_function, division

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import nsfg
import first
import thinkstats2
import thinkplot
import os
import pandas 
import statistics
df = pandas.read_csv('data.csv')


# In[2]:


from statistics import mode


# In[3]:


import seaborn as sns


# In[4]:


#data cleanup
df.drop(columns=['from_station_id','from_station_name','usertype','to_station_id','to_station_name','dpcapacity_end','events','dpcapacity_start'],axis=1)


# In[5]:


#for faster output
df2 = df[['tripduration','trip_id']]


# Here's the histogram of birth weights:

# In[6]:


hist = thinkstats2.Hist(df2.tripduration, label='Duration_mins')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='Time Duration (minutes)', ylabel='Count')


# Normalize the distribution

# In[7]:


n = hist.Total()
pmf = hist.Copy()
for x, freq in hist.Items():
    pmf[x] = freq / n


# Probability Mass Function (PMF).

# In[8]:


thinkplot.Hist(pmf)
thinkplot.Config(xlabel='Time Duration (minutes)', ylabel='PMF')


# In[9]:


#OUTLIERS
for pmf, freq in hist.Smallest(10):
    print(pmf, freq)
    
#10 lowest values are roughly 2 minutes. Could be people commuting to the train 


# In[10]:


for pmf, freq in hist.Largest(10):
    print(pmf, freq)
#10 largest values are close to an hour, most likely due to the fact that divvy bikes max out at an hour for single usage


# In[11]:


#Outliers test for trip duration
#There are outliers for 30 minutes and on 
sns.boxplot(x=df['tripduration'])


# PMF for each of the variables

# In[12]:


pmf = thinkstats2.Pmf(df.temperature, label='temperature')


# In[13]:


thinkplot.Hist(pmf)
thinkplot.Config(xlabel='temperature (F)', ylabel='Pmf')


# Stepped function PMF for Temperature

# In[14]:


thinkplot.Pmf(pmf)
thinkplot.Config(xlabel='Temperature (F)', ylabel='Pmf')


# Distributions for month, week,day and hour

# In[15]:


timeduration_pmf = thinkstats2.Pmf(df.tripduration, label = 'trip duration')
temperature_pmf = thinkstats2.Pmf(df.temperature, label = 'temperature')
month_pmf = thinkstats2.Pmf(df.month, label='month')
week_pmf = thinkstats2.Pmf(df.week, label='week')
day_pmf = thinkstats2.Pmf(df.day, label='day')
hour_pmf = thinkstats2.Pmf(df.hour, label='hour')


# In[16]:


sns.boxplot(x=df['temperature'])
#20 and below are shown as outliers, but Chicago winters do get very cold so I will keep the data


# Month Histogram

# In[17]:


thinkplot.Hist(month_pmf)
thinkplot.Config(xlabel='Months', ylabel='Pmf')


# In[18]:


sns.boxplot(x=df['month']) #No outliers for months


# Week Histogram

# In[19]:


thinkplot.Hist(week_pmf)
thinkplot.Config(xlabel='Weeks', ylabel='Pmf')


# In[20]:


sns.boxplot(x=df['week']) #no outliers in weeks


# Days Histogram

# In[21]:


thinkplot.Hist(day_pmf)
thinkplot.Config(xlabel='Days', ylabel='Pmf')


# In[22]:


sns.boxplot(x=df['day']) #no outliers in day


# Hours Histogram

# In[23]:


thinkplot.Hist(hour_pmf)
thinkplot.Config(xlabel='hours', ylabel='Pmf')


# In[24]:


sns.boxplot(x=df['hour']) #no outlier in hours


# Exploratory Analysis of the Data

# In[25]:


width=0.45
axis = [27, 46, 0, 0.6]
thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(timeduration_pmf, align='right', width=width)
thinkplot.Hist(temperature_pmf, align='left', width=width)
thinkplot.Config(xlabel='Time Duration comparison against Temperature)', ylabel='PMF', axis=axis)

thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([timeduration_pmf, temperature_pmf])
thinkplot.Config(xlabel='Point in Time duration (temperature)', axis=axis)


# Additional Exploratory Analysis

# In[26]:


df.temperature.mean() #62.999
df.tripduration.mean() #11.44
mode(df.month) #8, People ride the most in August
mode(df.week) #30 #People rode the most the 30th week of the month
mode(df.day) #1, people rode the most on the first of the month
mode(df.hour) #17, people mostly ride at 5PM


# Standard Deviation

# In[27]:


temp_std = df.temperature.std()
print(temp_std)


# In[28]:


tripduration_std = df.tripduration.std()
print(tripduration_std)


# Variance

# In[29]:


temp_var = df.temperature.var()
print(temp_var)


# In[30]:


tripduration_var = df.tripduration.var()
print(tripduration_var)


# Cumulative Distributive Function

# In[31]:


# Create variable with TRUE if temperature >= 70
warm = df['temperature'] >= 70

# Create variable with TRUE if temperature < 70
chilly = df['temperature'] < 70


warm_df = df[warm]
chilly_df = df[chilly]


# In[32]:


warm_tripduration_cdf = thinkstats2.Cdf(warm_df.tripduration, label='warm trip duration')
thinkplot.Cdf(warm_tripduration_cdf)
thinkplot.Config(xlabel='trip duration (minutes)', ylabel='CDF', loc='upper left')


# In[33]:


chilly_tripduration_cdf = thinkstats2.Cdf(chilly_df.tripduration, label='chilly trip duration')
thinkplot.Cdf(chilly_tripduration_cdf)
thinkplot.Config(xlabel='trip duration (minutes)', ylabel='CDF', loc='upper left')


# In[34]:


#comparison
thinkplot.Cdfs([chilly_tripduration_cdf,warm_tripduration_cdf])
thinkplot.Show(xlabel='trip duration (minutes)',ylabel='CDF')


# By comparing colder and warmer temperatures with their duration, we can see that chilly bike rides are slightly shorter than warmer bike rides

# Analytical Distribution

# In[35]:


#NORMAL CDF to for visual
thinkplot.PrePlot(3)

mus = [1.0, 2.0, 3.0]
sigmas = [0.5, 0.4, 0.3]
for mu, sigma in zip(mus, sigmas):
    xs, ps = thinkstats2.RenderNormalCdf(mu=mu, sigma=sigma, 
                                               low=-1.0, high=4.0)
    label = r'$\mu=%g$, $\sigma=%g$' % (mu, sigma)
    thinkplot.Plot(xs, ps, label=label)

thinkplot.Config(title='Normal CDF', xlabel='x', ylabel='CDF',
                 loc='upper left')


# In[71]:



#OBSERVED CDF AND MODEL
# estimate parameters: trimming outliers yields a better fit
mu, var = thinkstats2.TrimmedMeanVar(df.tripduration, p=0.01)
print('Mean, Var', mu, var)
    
# plot the model
sigma = np.sqrt(var)
print('Sigma', sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)
thinkplot.Plot(xs, ps, label='model', color='0.6')

# plot the data
cdf = thinkstats2.Cdf(df.tripduration, label='data')

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf) 
thinkplot.Config(title='Trip Duration',
                 xlabel='Trip Duration (Minutes)',
                 ylabel='CDF')


# Model fits data except for in th left tail up to 15 minutes

# If data is from a normal distribution then the plot will be straight

# In[72]:


n = 1000
thinkplot.PrePlot(3) 

mus = [0, 1, 5]
sigmas = [1, 1, 2]

for mu, sigma in zip(mus, sigmas):
    sample = np.random.normal(mu, sigma, n)
    xs, ys = thinkstats2.NormalProbability(sample)
    label = '$\mu=%d$, $\sigma=%d$' % (mu, sigma)
    thinkplot.Plot(xs, ys, label=label)

thinkplot.Config(title='Normal probability plot',
                 xlabel='standard normal sample',
                 ylabel='sample values')


# Normal Probability Plot for trip duration

# In[73]:


mean, var = thinkstats2.TrimmedMeanVar(df.tripduration, p=0.01)
std = np.sqrt(var)

xs = [-4, 4]
fxs, fys = thinkstats2.FitLine(xs, mean, std)
thinkplot.Plot(fxs, fys, linewidth=4, color='0.8')

xs, ys = thinkstats2.NormalProbability(df.tripduration)
thinkplot.Plot(xs, ys, label='trip duration')

thinkplot.Config(title='Normal probability plot',
                 xlabel='Standard deviations from mean',
                 ylabel='Trip duration (mins)')


# The results cause me to not choose a normal probability plot

# In[80]:


tripduration = df.tripduration
#Testing for normal lognormal distribution
def MakeNormalModel(tripduration):
    """Plots a CDF with a Normal model.

    weights: sequence
    """
    cdf = thinkstats2.Cdf(tripduration, label='tripduration')

    mean, var = thinkstats2.TrimmedMeanVar(tripduration)
    std = np.sqrt(var)
    print('n, mean, std', len(tripduration), mean, std)

    xmin = mean - 4 * std
    xmax = mean + 4 * std

    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)
    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')
    thinkplot.Cdf(cdf)


# In[82]:


MakeNormalModel(tripduration)
thinkplot.Config(title='Adult weight, linear scale', xlabel='Minutes',
                 ylabel='CDF', loc='upper right')


# Scatter Plots

# In[85]:


thinkplot.Scatter(df.temperature, df.tripduration, alpha=.04)
thinkplot.Config(xlabel='Temperature',
                 ylabel='Trip Duration',
                 legend=False)


# In[86]:


thinkplot.Scatter(df.day, df.tripduration, alpha=0.04, s=10)
thinkplot.Config(xlabel='Day',
                 ylabel='Trip Duration (Minutes)',
                 legend=False)


# In[90]:


#Covariance
np.corrcoef(df.tripduration, df.temperature)
np.corrcoef(df.tripduration, df.day)


# In[93]:


#Pearson's Correlation is more robust
def SpearmanCorr(xs, ys):
    xs = pd.Series(xs)
    ys = pd.Series(ys)
    return xs.corr(ys, method='spearman')


# In[94]:


SpearmanCorr(df.tripduration, df.temperature)


# In[95]:


SpearmanCorr(df.tripduration, df.day)


# Hypothesis Testing

# In[99]:


#Test Correlation
class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


# In[97]:


class DivvyTest(HypothesisTest):

    def TestStatistic(self, data):
        df.tripduration, df.temperature = data
        test_stat = abs(df.tripduration - df.temperature)
        return test_stat

    def RunModel(self):
        df.tripduration, df.temperature = self.data
        n = df.tripduration + df.temperature
        sample = [random.choice('HT') for _ in range(n)]
        hist = thinkstats2.Hist(sample)
        data = hist['H'], hist['T']
        return data


# In[101]:


def RunTests(live, iters=1000):
    """Runs the tests from Chapter 9 with a subset of the data.

    live: DataFrame
    iters: how many iterations to run
    """
    n = len(df)
    chilly_df = df[chilly]
    warm_df = df[warm]
    chilly_df = df[chilly]

    # Test
    data = chilly_df.values, warm_df.values
    ht = DiffMeansPermute(data)
    p1 = ht.PValue(iters=iters)

    ht = CorrelationPermute(data)
    p3 = ht.PValue(iters=iters)

    # chi-squared
    data = chilly_df.values, warm_df.values
    ht = DivvyTest(data)
    p4 = ht.PValue(iters=iters)

    print('%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f' % (n, p1, p2, p3, p4))


# Regression Test

# In[103]:


import statsmodels.formula.api as smf


# In[106]:


formula = 'tripduration ~ temperature'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()

