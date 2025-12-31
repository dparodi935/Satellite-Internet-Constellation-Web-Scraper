from bs4 import BeautifulSoup
import requests as r
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 

#%% Functions

def get_dataframe(url):
    headers = {'User-Agent': 'SatelliteGrowthBot/1.0 (https://github.com/dparodi935; dparodi935@gmail.com) Python-Requests/2.31.0'}
    response = r.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    launches_table = soup.find_all('table',{'class':'wikitable'})

    dfs = pd.read_html(str(launches_table)) # convert all tables to dataframes
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
    df = df[wanted_column_names]

    #rename headers
    df.columns = ['Launch Date','Number', 'Outcome']
        
    #drop planned launches
    df = df[df['Outcome'] != 'Planned']
    
    df['Launch Date'] = pd.to_datetime(df['Launch Date'], format='mixed', dayfirst=True, errors='coerce')
    df['Launch Date'] = df['Launch Date'].dt.normalize() #sets time to zero
    
    return df


def return_cumulative_data(df, do_log=False, dateshift=False):
    #this adds together launches that occurred on the same day
    cleaned_df = df.groupby(['Launch Date', 'Constellation'])['Number'].sum().reset_index()
    cleaned_df = cleaned_df.sort_values(by=['Launch Date'])
    
    #this finds the cumulative sum over time of the satellite launches
    cleaned_df['Cumulative'] = cleaned_df.groupby('Constellation')['Number'].cumsum()
    if do_log:  cleaned_df['Cumulative'] = np.log10(cleaned_df['Cumulative'])
    
    if dateshift:
        #aligning the times so they are counted relative to each constellation's first launch
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
    
    print(cleaned_df.sample(5))
    
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
url = 'https://en.wikipedia.org/wiki/List_of_Starlink_and_Starshield_launches'
df = get_dataframe(url)

wanted_column_names = ['Launch date, time (UTC)','Deployed','Outcome']
df = basic_cleaning(df, wanted_column_names)

df = df[df['Launch Date'].dt.year > 2018]

data['Starlink'] = df.reset_index()


#%% Amazon Leo

url = 'https://en.wikipedia.org/wiki/Amazon_Leo'
df = get_dataframe(url)

wanted_column_names = ['Date and time (UTC)','Satellites','Launch status']
df = basic_cleaning(df, wanted_column_names)

df = df[df['Launch Date'].dt.year > 2023]

data['Amazon Leo'] = df.reset_index()


#%% Qianfan
url = 'https://en.wikipedia.org/wiki/Qianfan'
df = get_dataframe(url)

wanted_column_names = ['Launch (UTC)','Name & number of satellites', 'Status']
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



url = 'https://en.wikipedia.org/wiki/Guowang'
df = get_dataframe(url)

wanted_column_names = ['Launch (UTC)','Number of satellites','Status']
df = basic_cleaning(df, wanted_column_names)


data['Guowang'] = df.reset_index()


#%% Plot
for constellation in data:
    data[constellation]['Number'] = pd.to_numeric(data[constellation]['Number'])
    data[constellation]['Constellation'] = constellation
    
merged_df = pd.concat(data.values(), ignore_index=True)
merged_df = merged_df.sort_values(by=['Launch Date']) 
#merged df is a combined, ordered list of all constellation launches

#%% Actual time area plot over all time

plot_data = return_cumulative_data(merged_df)
plot_data.plot.area(figsize=(12,6))

plt.xlim(plot_data.index[0],plot_data.index[-1])
plt.ylim(0,None)
plt.ylabel("Cumulative Number of Satellites")

plt.title("Cumulative number of satellites")

plt.show()

#%% Date shift line plot over all time

plot_data = return_cumulative_data(merged_df, dateshift=True)
plot_data.plot(figsize=(12,6))

plt.xlim(0,None)
plt.ylim(0,None)
plt.ylabel("Cumulative Number of Satellites")

plt.title("Cumulative number of satellites")

plt.show()


#%% Date shift line plot over first n years

plot_data = return_cumulative_data(merged_df, dateshift=True)
plot_data.plot(figsize=(12,6))

max_days = plot_data.drop(columns=['Starlink']).idxmax()
final_max_day = max_days.max()
n_of_years = math.ceil(final_max_day/365)
upper_bound_index = plot_data.index[np.argmin(abs(plot_data.index-365*n_of_years))]

plt.xlim(0,365*n_of_years)
plt.ylim(0,plot_data['Starlink'][upper_bound_index]*1.1)
plt.ylabel("Cumulative Number of Satellites")

plt.title(f"Cumulative number of satellites in the first {n_of_years} years")

plt.show()

#%% Date shift log line plot over all time
log_plot_data = return_cumulative_data(merged_df, do_log=True, dateshift=True)
log_plot_data.plot(figsize=(12,6))

plt.xlim(0,None)
plt.ylabel("Log of Cumulative Number of Satellites")

plt.title("Log of Cumulative Number of Satellites")

plt.show()
