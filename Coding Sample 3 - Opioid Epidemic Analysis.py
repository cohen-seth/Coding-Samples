# Script 1: state_od_count.py
# State Overdose Data

import pandas as pd

od_count = pd.read_csv('https://query.data.world/s/sobpxqd6thmlycbu6lxpfyvnx6xwzb')
print(od_count.columns)

state = od_count['State']
year = od_count['Year']
deaths = od_count['Data Value']

print(year)
od_count_grouped = od_count.groupby(['State'])
od_count_groupsum = od_count.groupby(['Year'])
print(od_count_groupsum)
deaths_sum_byyear = od_count_groupsum('Data Value').sum()
print(deaths_sum_byyear)


########################################################################################
## Script 2:ma_towns.py
## Data = MA towns
## Animation with scatter plot where size corresponds to deaths


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib import ticker


data = pd.read_csv('https://query.data.world/s/5n67ctfovqgs2cdwjzxbks2bagumru')
print(data)
print(data.columns)


lat = data['latitude']
lon = data['longitude']

# Rate and Counts ~ 5 year intervals
city = data['Municipality']
count01 = data['Confirmed Opioid Related Death Count 2001-2005']
rate01 = data['Annual Opioid-Related Death Rate per 100,000 People from 2001-2005']

count06 = data['Confirmed Opioid Related Death Count 2006-2010']
rate06 = data['Annual Opioid-Related Death Rate  per 100,000 People from 2006-2010']

count11 = data['Confirmed Opioid Related Death Count 2011-2015']
rate11 = data['Confirmed Opioid Related Death Count 2011-2015']


lat = data['latitude'].values
lon = data['longitude'].values


## Plot: Animation

fig = plt.figure(figsize=(8, 8))

## Plotting basemap centered at (42,-71.5) in order to view MA
m = Basemap(projection='lcc', resolution='h', lat_0= 42, lon_0= -71.5, width=.3E6, height=.3E6)

m.shadedrelief()
m.drawcounties(color='k')
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

## Plotting a marker for Boston, MA
Boston_lat = 42.36
Boston_lon = -71.05
Bx, By, = m(Boston_lon, Boston_lat)
m.scatter(Bx, By, color = 'yellow', marker = '*', label = 'Boston, MA')

## getting the right lat and long
x1, y1 = m(lon,lat)

## scatter city data, with color reflecting population and size reflecting area
cf = m.scatter(x1, y1, c=np.log(count01), s = count01, marker = 'o', cmap='Reds', alpha = 0.5)

## colorbar and legend
cb = plt.colorbar(cf, orientation = 'horizontal')
cb.set_label('$\ln(Death Count)$')
plt.clim(0,5)

plt.legend(scatterpoints=1, frameon=False,labelspacing=1, loc='lower right')
plt.title('Opioid Deaths by MA City 2001-05')
fig.savefig('plot01.png')
plt.close()

fig2 = plt.figure(figsize=(8, 8))

m = Basemap(projection='lcc', resolution='h', lat_0= 42, lon_0= -71.5, width=.3E6, height=.3E6)

Boston_lat = 42.36
Boston_lon = -71.05
Bx, By, = m(Boston_lon, Boston_lat)
m.scatter(Bx, By, color = 'yellow', marker = '*', label = 'Boston, MA')

m.shadedrelief()
m.drawcounties(color='k')
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

m.scatter(x1, y1, c=np.log(count06), s = count06, marker = 'o', cmap='Reds', alpha = 0.5)

cb = plt.colorbar(cf, orientation = 'horizontal')
cb.set_label('$\ln(Death Count)$')
plt.clim(0,5)
plt.legend(scatterpoints=1, frameon=False,labelspacing=1, loc='lower right')
plt.title('Opioid Deaths by MA City 2006-11')
fig2.savefig('plot02.png')
plt.close()
#plt.show()


## Count11
fig3 = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', lat_0= 42, lon_0= -71.5, width=.3E6, height=.3E6)

Boston_lat = 42.36
Boston_lon = -71.05
Bx, By, = m(Boston_lon, Boston_lat)
m.scatter(Bx, By, color = 'yellow', marker = '*', label = 'Boston, MA')


m.shadedrelief()
m.drawcounties()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

m.scatter(x1, y1, c= np.log(count11), s = count11, marker = 'o', cmap='Reds', alpha = 0.5)

cb = plt.colorbar(cf, orientation = 'horizontal')
cb.set_label('$\ln(Death Count)$')
plt.clim(0,5)

plt.legend(scatterpoints=1, frameon=False,labelspacing=1, loc='lower right')
plt.title('Opioid Deaths by MA City 2011-15')
fig3.savefig('plot03.png')
plt.close()
#plt.show

########################################################################################
## Script 3:gender.py
## Gender & Age Stats

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

data = pd.read_csv('https://query.data.world/s/ipctxsbgw2lfrerhg3xnxhdfq5ji2p')
print(data.columns)

gender = data['Gender']
age = data['Age Range']
deaths = data['Number of Opioid-Related Deaths from 2013-2014']
percent = data['Percent of Opioid-Related Death among all Deaths from 2013-2014']
annual = data['Annual Opiod-Related Death Rate per 100,000 People from 2013-2014']

## Coding the portion of the data which corresponds to female data
female = gender[::2]
age_f = age[::2]
deaths_f = deaths[::2]
percent_f = percent[::2]
annual_f = annual[::2]

#male
male = gender[1::2]
age_m = age[1::2]
deaths_m = deaths[1::2]
percent_m = percent[1::2]
annual_m = annual[1::2]
#print(male)

## Bar Graph 1: Comparing deaths (by age) between the genders
fig = plt.figure(figsize = [9,12])

## Stacked bar graph
plt.bar(age_m, deaths_m, color = 'k', label = 'Male Deaths')
plt.bar(age_f, deaths_f, color = 'g', label = 'Female Deaths')

plt.grid(axis = 'y', ls = '--')
plt.legend()
plt.xlabel('Age Range')
plt.ylabel('Number of Opioid Related Deaths')
plt.title('Male vs. Female Deaths by Age,\n 2013-2014')
fig.savefig('Bar: Death Count.png')
plt.close()

## Bar Graph 2: Looking at the percentage of deaths by age and by gender
fig1 = plt.figure(figsize = [9,12])
plt.bar(age, percent)
plt.bar(gender, percent)
plt.ylabel('Percent')
plt.title('Percentage of Opioid Related Deaths: Age & Gender \n 2013-2014')
plt.grid(axis='y', ls ='--')
fig1.savefig('Bar:Percent.png')
plt.close()


## Pie Chart: Similar to the bar graphs, showing the percentage of the total that each age range (sorted for gender) accounts for
fig2= plt.figure(figsize = [9,12])
ax = fig2.add_subplot(111)


labels = ('Female:18-24', 'Male:18-24', 'Female:25-34', 'Male 25-34', 'Female:35-49', 'Male:35-49', 'Female:50-64', 'Male:50-64', '','')

## Explode: maked the two largest sections seperate out from the rest of the graph
pie_wedge_collection = ax.pie(percent, explode = (0, 0, 0.2, 0.2, 0, 0,0,0,0,0), labels=labels, labeldistance=1.05)

for pie_wedge in pie_wedge_collection[0]:
    pie_wedge.set_edgecolor('black')

plt.xlabel('*Male & Female: 65+ = 0.4%')
plt.title('Percentage of Deaths by Age and Gender, \n 2013-2014')
fig2.savefig('Pie: percentage.png')
plt.close()


########################################################################################
## Script 4:  adjusted_opioid_deathrate.py
## Data from Data.world
## Age Adjusted Opioid Death Rate per 100,000 - MA

import numpy as np
import matplotlib.pyplot as plt
import MyFunctions as mf
import pandas as pd


## Data Import
## 1st file: Contains opioid data specific to Massachusetts
## 2nd file: Contains GDP per Capita data on the entire United States
ma_dr = pd.read_csv('https://query.data.world/s/dm5kcjpf5flarwyrexesgwr2v3rfai')
gdp = pd.read_excel('https://query.data.world/s/imhuhdnl4f2m4b4mxnijmlwxj324l3', index_col=0)

gdp = gdp.T
MA = gdp['Massachusetts']
MA = MA[3:-1]
US = gdp['United States']
US = US[3:-1]
print(US)
print(MA)


ma_dr = ma_dr.drop('Geography', 1)
ma_dr = ma_dr.drop(ma_dr.index[1])
ma_dr = ma_dr.drop(ma_dr.index[-1])
print(ma_dr)

year = ma_dr['Year']
year_MA = year[1:-1:2]
year_US = year[2:-1:2]
print(year_MA)

## Assiging the death rate data to death_rate and changing it to a numeric value
death_rate = ma_dr['Age-Adjusted Opioid-Related Death Rate per 100,000 People']
death_rate = pd.to_numeric(death_rate)
death_rateMA = death_rate[1:-1:2]
death_rateUS = death_rate[2:-1:2]
print(death_rateMA)

## Fig: Plotting the year vs GDP per capita and year vs Deaths for MA
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(year_MA, MA, label = 'MA: GDP per capita', color = 'b')
ax2 = ax.twinx()
ax2.plot(year_MA, death_rateMA, label = 'Deaths per 100,000', color = 'r')
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel('GDP per capita')
ax2.set_ylabel('Deaths')
ax.set_ylim(54000, 63000)
ax2.set_ylim(0 ,25)
ax.legend(loc=3)
ax2.legend(loc=4)
plt.title('MA: Deaths per 100k and GDP per Capita vs. Year')
plt.savefig('GDP v Deaths: MA.png')
plt.close()

## Fig1: Plotting year vs GDP and year vs Deaths for US
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(year_US, US, label = 'US: GDP per capita', color = 'b')
ax2 = ax.twinx()
ax2.plot(year_US, death_rateUS, label = 'Deaths per 100,000', color = 'r')
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel('GDP per capita')
ax2.set_ylabel('Deaths')
ax.set_ylim(44000, 50000)
ax2.set_ylim(0 ,25)
ax.legend(loc=3)
ax2.legend(loc=4)
plt.title('United States: Deaths per 100k and GDP per Capita vs. Year')
plt.savefig('GDP v Deaths:US.png')
plt.close()

## Fig2: Plotting US & MA death rates together
plt.plot(year_MA, death_rateMA, color = 'b', label = 'MA')
plt.plot(year_US, death_rateUS, color = 'c', label = 'US')
plt.title('Age-Adjusted Opioid-Related Death Rate per 100,000 People:\n United States v. Massachusettes')
plt.xticks(range(2000,2026), rotation = 45)
plt.xlabel('Year')
plt.ylabel('Death Rate per 100,000 people')
plt.grid(True)
plt.legend(loc=0)

## Creating numpy arrays from the previous data which was imported using pandas
## In order to create linear/quadratic fits and projects through the year 2030
us_fit = np.array(death_rateUS)
print(us_fit)
deathrate_fit = np.array([5.8, 7.7, 8.0, 9.4, 7.9, 8.8, 10.0, 9.7, 9.4, 9.4, 8.5, 9.9, 11.1, 14.4, 19.9])
print(len(deathrate_fit))
yearfit = np.linspace(2000, 2014, 15)

## Linear Fit: United States
coef_us = np.polyfit(yearfit,us_fit, 1)
print(coef_us)

# Generating a function with the coefficients from above
uslfit = np.poly1d(coef_us)
# Generating an array of x values from the first year until the last with 100 points
usxfit = np.linspace(yearfit[1], yearfit[-1], 100)
# Generating y values out of the function lfit
usyfit_1 = uslfit(usxfit)

#plt.plot(usxfit, usyfit_1, color = 'yellow', label = 'Linear Fit')
usxfit_p = np.linspace(yearfit[0], yearfit[-1]+12, 100)
usyfit_1p = uslfit(usxfit_p)
plt.plot(usxfit_p, usyfit_1p, ls='--', color ='c', label = 'US: Linear Fit Projection')


## Linear Fit: MA

####### Running Mean for Death Rate MA

MA_rm, window = mf.running_mean(deathrate_fit, 5)
print MA_rm
plt.plot(yearfit[2:-2], MA_rm[2:-2], label = '5 Year Running Mean', color = 'blue')

coef_lin_MA = np.polyfit(yearfit,deathrate_fit, 1)
print(coef_lin_MA)

## Generating a function with the coefficients from above
lfit = np.poly1d(coef_lin_MA)
## Generating an array of x values from the first year until the last with 100 points
xfit = np.linspace(yearfit[1], yearfit[-1], 100)
## Generating y values out of the function lfit
yfit_1 = lfit(xfit)

## Linear Fit Projections up to 2030
xfit_p = np.linspace(yearfit[0], yearfit[-1]+12, 100)
yfit_1p = lfit(xfit_p)
plt.plot(xfit_p, yfit_1p, ls='--', color ='r', label = 'MA: Linear Fit Projection')

## Quadrati Fit for MA
coef_quad_MA = np.polyfit(yearfit,deathrate_fit, 2)
print(coef_quad_MA)

## Generating a function with the coefficients from above
qfit = np.poly1d(coef_quad_MA)
## Generating an array of x values from the first year until the last with 100 points
xfit = np.linspace(yearfit[1], yearfit[-1], 100)
## Generating y values out of the function lfit
yfit_2 = qfit(xfit)

## Linear Fit Projections up to 2030
xfit_p = np.linspace(yearfit[0], yearfit[-1]+12, 100)
yfit_2p = qfit(xfit_p)
plt.plot(xfit_p, yfit_2p, ls='--', color ='k', label = 'MA: Quadratic Fit Projection')

plt.legend()
plt.savefig('Deaths Lin Fit Proj.png')
#plt.show()
plt.close()