# -*- coding: utf-8 -*-
"""
@author: Rick BOURGET
"""
import csv
import datetime
import re
import codecs
import requests
import pandas as pd
import numpy as np
import cufflinks as cf
import os
import pylab
from plotly.offline import download_plotlyjs, plot, iplot,init_notebook_mode
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
init_notebook_mode(connected=True)

def get_google_finance_intraday(ticker, period=60, days=1, exchange='EPA'):
 
    url = 'https://finance.google.com/finance/getprices' + \
          '?p={days}d&f=d,o,h,l,c,v&q={ticker}&i={period}&x={exchange}'.format(ticker=ticker, 
                                                                               period=period, 
                                                                               days=days,
                                                                               exchange=exchange)
    
    page = requests.get(url)
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    
    columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    rows = []
    times = []
    for row in reader:
        if re.match(r'^[a\d]', row[0]): 
            if row[0].startswith('a'): 
                start = datetime.datetime.fromtimestamp(int(row[0][1:])) #en format unix stamp
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=period*int(row[0]))) #conversion unix -> format temps classique
            rows.append(map(float, row[1:]))
    
    if len(rows):
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'), columns=columns)
    else:
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))
    
#=================================================Graph================================================================

def rsiFunc(prices, n=14): #indicator if a product is looking to be overbought (over 70) or oversold (under 30)
    deltas = np.diff(prices) #liste de differences entre nos prix
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n    #liste des valeurs haussieres
    down = -seed[seed < 0].sum()/n  #liste des valeurs baissieres
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100 - 100/(1+rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (up*(n-1)+upval)/n
        down = (down*(n-1)+downval)/n
        rs = up/down
        rsi[i] = 100 - 100/(1.+rs)
    return rsi


def movingAverage(values,window):#values = data, window=range of the MA
    weights = np.repeat(1.0,window)/window
    smas = np.convolve(values,weights,'valid') #on somme nos poids ponderes avec toutes nos valeurs
    return smas #returns a list as a numpy array

#exponential moving average places more weight on more recent datas so it makes it react quicker
def ExpMovingAverage(values,window):
    weights = np.exp(np.linspace(-1,0,window)) #on distribue equitablement des valeurs entre exp(-1) et 0
    weights /= weights.sum()                   #pareil que weights = weights / weights.sum(), pour chaque valeur de l'array, elle sera divisee par la somme des poids totaux
    ema = np.convolve(values,weights,mode = 'full')[:len(values)]
    ema[:window] = ema[window]
    return ema

def computeMACD(x,slow=26,fast=12):
    '''
    macd line = 12ema - 26ema
    signal line = 9ema of the macd line
    histogram = macd line - signal line
    '''
    emaslow = ExpMovingAverage(x,slow)
    emafast = ExpMovingAverage(x,fast)
    return emaslow,emafast,emafast - emaslow
    

def graphData(stock, MA1, MA2, MA3):
    
    #df=pd.read_csv(stock+'.csv',index_col=0)
    df=get_google_finance_intraday(stock)
    
    startingPoint=len(df['Close'][MA2-1:]) #pour avoir un graphe propre et sans coupure, je commence avec la moyenne mobile la plus importante en periode
    
    #=========================Bougies + Volume==============================
    ax1 = go.Candlestick(x=df.index[-startingPoint:],
                        open=df['Open'][-startingPoint:],
                        high=df['High'][-startingPoint:],
                        low=df['Low'][-startingPoint:],
                        close=df['Close'][-startingPoint:],
                        name=stock+' Prices')
    
    axV= go.Scatter(x=df.index[-startingPoint:],
                   y=df['Volume'][-startingPoint:],
                   yaxis='y5',
                   fill='tozeroy',
                   mode='none',
                   name='Volume',
                   line=dict(color='#00ffe8'))
    
    #========================indicateurs===========================
    average1 = movingAverage(df['Close'],MA1)
    average2 = movingAverage(df['Close'],MA2)
    average3 = movingAverage(df['Close'],MA3)
    
    ma1 = go.Scatter(x=df.index[-startingPoint:],
                     y=average1[-startingPoint:],
                     name='MA'+str(MA1))
    ma2 = go.Scatter(x=df.index[-startingPoint:],
                     y=average2[-startingPoint:],
                     name='MA'+str(MA2))
    ma3 = go.Scatter(x=df.index[-startingPoint:],
                     y=average3[-startingPoint:],
                     name='MA'+str(MA3))
    
    #============================RSI================================
    rsi = rsiFunc(df['Close'])
    
    axRSI=go.Scatter(x=df.index[-startingPoint:],
                    y=rsi[-startingPoint:],
                    yaxis='y2',
                    name='RSI')
    
    seuil70=np.repeat(70,len(df.index))
    seuil30=np.repeat(30,len(df.index))
    
    axRSI70=go.Scatter(x=df.index[-startingPoint:],
                      y=seuil70[-startingPoint:],
                      yaxis='y2',
                      mode='lines',
                      name='Seuil 70%',
                      showlegend=False)
    
    axRSI30=go.Scatter(x=df.index[-startingPoint:],
                      y=seuil30[-startingPoint:],
                      yaxis='y2',
                      mode='lines',
                      name='Seuil 30%',
                      showlegend=False)
    
    
    #===========================MACD================================
    emaslow,emafast,difference = computeMACD(df['Close'])
    
    axMACDslow=go.Scatter(x=df.index[-startingPoint:],
                          y=emaslow[-startingPoint:],
                          yaxis='y3',
                          showlegend=False,
                          name='Slow EMA')
    axMACDfast=go.Scatter(x=df.index[-startingPoint:],
                          y=emafast[-startingPoint:],
                          yaxis='y3',
                          showlegend=False,
                          name='Fast EMA')
    
    couleur=[]
    for i in range(len(difference)):
        if difference[i]>=difference[i-1]:
            couleur.append('#00ff00')
        elif difference[i]<difference[i-1]:
            couleur.append('#ff0000')
    df['couleur']=couleur
            
    
    
    axMACDbar=go.Bar(x=df.index[-startingPoint:],
                    y=difference[-startingPoint:],
                    yaxis='y4',
                    name='MACD Variation',
                    marker=dict(color=df['couleur'][-startingPoint:]),
                    opacity=0.3)
    
    
    layout=go.Layout(xaxis=dict(
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=2,
                            mirror='ticks',
                            rangeslider=dict(
                                visible=False)),
                     plot_bgcolor='#000000',
                    yaxis=dict(
                            color='#5998ff',
                            linewidth=2,
                            mirror='ticks',
                            domain=[0.2,0.8],
                            anchor='free',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=14,
                                color='black')),
                    yaxis2=dict(
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=3,
                            mirror='ticks',
                            domain=[0.8,1],
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=14,
                                color='black')),
                    yaxis3=dict(
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=3,
                            mirror='ticks',
                            domain=[0,0.2],
                            anchor='free',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=14,
                                color='black')),
                    yaxis4=dict(
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=3,
                            mirror=True,
                            domain=[0,0.2],
                            side='right',
                            overlaying='y3',
                            anchor='y3',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=14,
                                color='black')),
                    yaxis5=dict(
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=3,
                            mirror='ticks',
                            domain=[0.2,0.8],
                            side='right',
                            overlaying='y',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=1,
                                color='black'))
                    )
                           
    data=[ax1,ma1,ma2,ma3,axRSI,axRSI70,axRSI30,axMACDslow,axMACDfast,axMACDbar,axV]
    #fig = tls.make_subplots(rows=1,cols=1,subplot_titles ='Prices with MA'+MA1'+ MA'+MA2)
    fig = go.Figure(data=data,layout=layout)
    fig['layout'].update(title=stock+' prices pulled without csv registration')
    return fig

# input data
#ticker = input("Stock to pull :")
ticker = 'CA'
ticker = (ticker,) 
period = 60
days = 1
exchange = 'EPA'

config = {'showLink': False,'modeBarButtonsToRemove': ['sendDataToCloud']} #retirer le lien qui redirige vers plotly
for stock in ticker:
    fig=graphData(stock,5,8,13)
    py.plot(fig,config=config)
    
