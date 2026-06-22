from bs4 import BeautifulSoup
from datetime import date
import requests as r
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import math 
from io import StringIO

#%% Functions

def get_dataframe(url):
    headers = {'User-Agent': 'SatelliteGrowthBot/1.0 (https://github.com/dparodi935; dparodi935@gmail.com) Python-Requests/2.31.0'}
    response = r.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    launches_table = soup.find_all('table',{'class':'wikitable'})

    dfs = pd.read_html(StringIO(str(launches_table))) # convert all tables to dataframes
    df = dfs[0] #get first table on page
    
    #this removes the multi-column headings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
        
    return df

def basic_cleaning(df, wanted_column_names):
    #remove references
    df.replace(r'\[\d+\]','',regex=True,inplace=True)
    df.columns = df.columns.str.replace(r'\[\d+\]','',regex=True) #d+ means numbers

    #clean column names
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

    #drop columns I don't want
    chosen_wanted_column_names = []
    #this will select the actual column names by looking for appearance of the user input, e.g. UTC
    for name in wanted_column_names:
        for actual_name in list(df.columns):
            if name.lower() in actual_name.lower():
                chosen_wanted_column_names.append(actual_name)
    
    df = df[chosen_wanted_column_names]

    #rename headers
    df.columns = ['Launch Date','Number', 'Outcome']
        
    #drop planned and unsuccessful launches
    df = df[df['Outcome'] != 'Planned']
    df = df[df['Outcome'] != 'Failure']
    
    df['Launch Date'] = pd.to_datetime(df['Launch Date'], format='mixed', dayfirst=True, errors='coerce')
    df['Launch Date'] = df['Launch Date'].dt.normalize() #sets time to zero
    
    return df


def return_cumulative_data(df, do_log=False, dateshift=False):
    #add dummy launches so data stretches to present day
    constellations = list(df["Constellation"].unique())
    pd_today_date = pd.Timestamp(date.today())
    for c in constellations:
        df.loc[len(df)] = [0, pd_today_date, 0, "Success",c]    
    
    #this adds together launches that occurred on the same day
    cleaned_df = df.groupby(['Launch Date', 'Constellation'])['Number'].sum().reset_index()
    cleaned_df = cleaned_df.sort_values(by=['Launch Date'])
    
    #this finds the cumulative sum over time of the satellite launches
    cleaned_df['Cumulative'] = cleaned_df.groupby('Constellation')['Number'].cumsum()
    
    if do_log:  cleaned_df['Cumulative'] = np.log10(cleaned_df['Cumulative'])
    
    if dateshift:
        cleaned_df['Day Zero'] = cleaned_df.groupby('Constellation')['Launch Date'].transform('min')
        cleaned_df['Launch Date'] = (cleaned_df['Launch Date']-cleaned_df['Day Zero']).dt.days
    
    #data rearranged for plotting: now each constellation is its own column
    plot_data = cleaned_df.pivot(index='Launch Date',columns='Constellation',values='Cumulative')
    plot_data = plot_data.ffill()
    
    if dateshift:
        last_days = cleaned_df.groupby('Constellation')['Launch Date'].max()
        for constellation in plot_data.columns:
            plot_data.loc[plot_data.index > last_days[constellation], constellation] = np.nan

    return plot_data

def return_launch_data(df, do_log=False, dateshift=False):
    #this adds together launches that occurred on the same day
    #TO ADD: group launches by month
    cleaned_df = df.groupby(['Constellation', pd.Grouper(key='Launch Date', freq='M')])['Number'].sum().reset_index()
    cleaned_df = cleaned_df.sort_values(by=['Launch Date'])
        
    if dateshift:
        #aligning the times so they are counted relative to each constellation's first launch
        cleaned_df['Day Zero'] = cleaned_df.groupby('Constellation')['Launch Date'].transform('min')
        cleaned_df['Launch Date'] = (cleaned_df['Launch Date']-cleaned_df['Day Zero']).dt.days
    
    #data rearranged for plotting: now each constellation is its own column
    plot_data = cleaned_df.pivot(index='Launch Date',columns='Constellation',values='Number')
    plot_data = plot_data.ffill()
    
    if dateshift:
        last_days = cleaned_df.groupby('Constellation')['Launch Date'].max()
        for constellation in plot_data.columns:
            plot_data.loc[plot_data.index > last_days[constellation], constellation] = np.nan

    return plot_data


#%%
data = {}


#%% Starlink
print("Starlink")
url = 'https://en.wikipedia.org/wiki/List_of_Starlink_and_Starshield_launches'
df = get_dataframe(url)

wanted_column_names = ['(UTC)','Deployed','Outcome','Status']
df = basic_cleaning(df, wanted_column_names)

#remove prototype launch
df = df[df['Launch Date'].dt.year > 2018]

data['Starlink'] = df.reset_index()


#%% Amazon Leo
print("Amazon LEO")
url = 'https://en.wikipedia.org/wiki/Amazon_Leo'
df = get_dataframe(url)

wanted_column_names = ['UTC','Satellites','Outcome','Status']
df = basic_cleaning(df, wanted_column_names)

#remove prototype launch
df = df[df['Launch Date'].dt.year > 2023]

data['Amazon Leo'] = df.reset_index()


#%% Qianfan
print("Qianfan")
url = 'https://en.wikipedia.org/wiki/Qianfan'
df = get_dataframe(url)

wanted_column_names = ['UTC','Name & number of satellites','Outcome','Status']
df = basic_cleaning(df, wanted_column_names)

#Change satellite count from cumulative
df['Number'] = df['Number'].str.replace(r'\(.*?\)','', regex=True).str.strip()
df['Number'] = df['Number'].str.replace('Qianfan','').str.strip()
numbers = df['Number'].str.split('-', expand=True) # expand creates two separate columns
left = pd.to_numeric(numbers[0])
right = pd.to_numeric(numbers[1])
df['Number'] = right-left+1


data['Qianfan'] = df.reset_index()


#%% Guowang
print("Guowang")
url = 'https://en.wikipedia.org/wiki/Guowang'
df = get_dataframe(url)

wanted_column_names = ['UTC','Number of satellites','Outcome','Status']
df = basic_cleaning(df, wanted_column_names)


data['Guowang'] = df.reset_index()


#%% 
for constellation in data:
    data[constellation]['Number'] = pd.to_numeric(data[constellation]['Number'])
    data[constellation]['Constellation'] = constellation
    
merged_df = pd.concat(data.values(), ignore_index=True)
merged_df = merged_df.sort_values(by=['Launch Date']) 
#merged df is a combined, ordered (by time) list of all constellation launches


#%%Plot 
fig, ax = plt.subplots(2,2)
subplot_size = (12,6)


#%% Actual time area plot over all time

plot_data = return_cumulative_data(merged_df)
plot_data.plot.area(ax=ax[0][0],figsize=subplot_size)

ax[0][0].set_xlim(plot_data.index[0],plot_data.index[-1])
ax[0][0].set_ylim(0,None)

ax[0][0].set_title("Cumulative number of satellites")


#%% Date shift line plot over all time

plot_data = return_cumulative_data(merged_df, dateshift=True)
plot_data.plot(ax=ax[0][1],figsize=subplot_size )

ax[0][1].set_xlim(0,None)
ax[0][1].set_ylim(0,None)

ax[0][1].set_title("Cumulative number of satellites")


#%% Date shift line plot over first n years

plot_data = return_cumulative_data(merged_df, dateshift=True)
plot_data.plot(ax=ax[1][0],figsize=subplot_size)

max_days = plot_data.drop(columns=['Starlink']).idxmax()
final_max_day = max_days.max()
n_of_years = math.ceil(final_max_day/365)
upper_bound_index = plot_data.index[np.argmin(abs(plot_data.index-365*n_of_years))]

ax[1][0].set_xlim(0,365*n_of_years)
ax[1][0].set_ylim(0,plot_data['Starlink'][upper_bound_index]*1.1)

ax[1][0].set_title(f"Cumulative number of satellites in the first {n_of_years} years")


#%% Date shift log line plot over all time
log_plot_data = return_cumulative_data(merged_df, do_log=True, dateshift=True)
log_plot_data.plot(ax=ax[1][1],figsize=subplot_size )

ax[1][1].set_xlim(0,None)

ax[1][1].set_title("Log of Cumulative Number of Satellites")


#%% In text, put the latest total number for each constellation
fig.tight_layout(rect=[0, 0.05, 1, 1])

cumulative_df = return_cumulative_data(merged_df)
total_numbers_dict = dict(cumulative_df.iloc[-1])
box = {"fill":False}

number_of_cons = len(total_numbers_dict.keys())
for c in range(number_of_cons):
    line = list(total_numbers_dict.items())[c]
    fig.text((c+0.25)*(1/number_of_cons),0.02,f"{line[0]}: {int(line[1])}")

#%% Divider
divider = plt.Line2D([0, 1], [0.05, 0.05], transform=fig.transFigure, color='black')
fig.add_artist(divider)

#%%
plt.show()
