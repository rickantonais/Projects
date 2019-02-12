import csv
import datetime
import re
import codecs
import requests
import pandas as pd
import numpy as np


#==================================Strategies + signals =======================
def investStrategy(df):
    #========definition des 2 vecteurs pour obtenir la matrice de rendement====
    poids=np.asarray([df['ordres']])
    log_ret=np.asarray([np.log(df['Close']).diff()])
    #=====================matrice de rendement=================================
    rendement_mat=poids.transpose().dot(log_ret)
    rendements=pd.Series(np.diag(rendement_mat),index=df.index) #diagonal = returns
    return rendements
    

def signauxMACD(signal,LineMACD):
    diff = np.subtract(LineMACD,signal)
    signal=[]
    for i in range(len(diff)):
        if diff[i]>0.0001:
            signal.append(1) #buy
        elif diff[i]<=0.0001:
            signal.append(-1) #sell
    diff=signal
    return diff

def signauxRSI(df_rsi):
    signal=[]
    for i in range(len(df_rsi)):
        if (df_rsi[i] - 30)<=0:
            signal.append(1) #buy
        elif (df_rsi[i]-70)>=0:
            signal.append(-1) #sell
        else:
            signal.append(0)
    return signal
        
def signauxEMA(EMA1,EMA2,EMA3):
    signal=[]
    for i in range(len(EMA1)):
        if EMA1[i]>EMA2[i]>EMA3[i]:
            signal.append(1)
        elif EMA1[i]<EMA2[i]<EMA3[i]:
            signal.append(-1)
        else:
            signal.append(0)
    return signal

def shortLongStrategy(signal1,signal2,signal3):
    signal=[]
    for i in range(len(signal1)):
        if signal1[i] == 1 and (signal2[i] == 1 or signal3[i] == 1):
            signal.append(1)
        elif signal1[i] == 1 and signal2[i] == 0 and signal3[i] ==0:
            signal.append(0)
        elif signal1[i]==-1 and (signal2[i] == -1 or signal3[i] == -1):
            signal.append(-1)
        elif signal1[i] == -1 and signal2[i] == 0 and signal3[i] == 0:
            signal.append(0)
        else:
            signal.append(0)
    return signal

def buffer(df):
    buffer=[]
    previous=0
    for row in df:
        if (row!=0):
            previous=row
            buffer.append(row)
        elif(row==0):
            buffer.append(previous)
    return buffer

def contrat(buff):
    final=[]
    previous=0
    for x in range(len(buff)):
        if(buff[x]==1):
            if(previous==1):
                final.append(0)
            else:
                final.append(1)
                previous=1
        if(buff[x]==-1):
            if(previous==-1):
                final.append(0)
            else:
                final.append(-1)
                previous=-1
        if(buff[x]==0):
            final.append(0)
    return final
            
def backtestPerformance(df):
    log_returns = np.log(df).diff()
    cum_log_returns = log_returns.cumsum()            #cumulative log returns
    total_rel_returns=100*(np.exp(cum_log_returns)-1) #total relative returns
    return log_returns,cum_log_returns,total_rel_returns

#========================index google scrap====================================
def get_index_google_finance_intraday(ticker, period=60, days=2):
 
    # build url
    url = 'https://finance.google.com/finance/getprices' + \
          '?p={days}d&f=d,o,h,l,c,v&q={ticker}&i={period}'.format(ticker=ticker, 
                                                                               period=period, 
                                                                               days=days)
    page = requests.get(url)
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    
    columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    rows = []
    times = []
    for row in reader:
        if re.match(r'^[a\d]', row[0]): #[] pour le faire sur toute la colonne, sinon il s'arrete au premier, ^ pour le debut de la chaine de caractere, le caractere a et \d designe un nombre decimale
            if row[0].startswith('a'): #si le regex precedent a ete rencontre, alors ...
                start = datetime.datetime.fromtimestamp(int(row[0][1:])) #en format unix stamp
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=period*int(row[0]))) #conversion unix -> format temps classique
            rows.append(map(float, row[1:]))
    
    if len(rows):
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'), columns=columns)
    else:
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))

#==========================stocks google scrap=================================
def get_google_finance_intraday(ticker, period=60, days=2, exchange='NASD'):
 
    # build url
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
        if re.match(r'^[a\d]', row[0]): #[] pour le faire sur toute la colonne, sinon il s'arrete au premier, ^ pour le debut de la chaine de caractere, le caractere a et \d designe un nombre decimale
            if row[0].startswith('a'): #si le regex precedent a ete rencontre, alors ...
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

import pylab
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import plotly.figure_factory as ff

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
    #pour le calcul de l'ema, on va limiter a la longueur de la dataframe en entree pour avoir une liste qui soit de meme taille que la dataframe
    ema = np.convolve(values,weights,mode = 'full')[:len(values)]
    ema[:window] = ema[window] #les "window" premieres valeures seront egales a la "window"-ieme valeur
    return ema

def computeMACD(x,slow=10,fast=3): 
    #par defaut (fast=12,slow=26,signal=9)
    #pour intraday(fast=3,slow=10,signal=16)
    '''
    macd line = 12ema - 26ema
    signal line = 9ema of the macd line
    histogram = macd line - signal line
    '''
    emaslow = ExpMovingAverage(x,slow)
    emafast = ExpMovingAverage(x,fast)
    return emaslow,emafast,emafast - emaslow
    

def graphStockData(stock, MA1, MA2, MA3):
    
    df=get_google_finance_intraday(stock)
    dowJonesIndex = get_index_google_finance_intraday('.DJI')
    nasdaqIndex=get_index_google_finance_intraday('.IXIC')
    benchDOW=np.log(dowJonesIndex['Close']).diff().cumsum()
    benchNASDQ=np.log(nasdaqIndex['Close']).diff().cumsum()
    
    startingPoint=len(df['Close'][MA3-1:])
    
    #========================indicateurs===========================
    EMaverage1 = ExpMovingAverage(df['Close'],MA1)
    EMaverage2 = ExpMovingAverage(df['Close'],MA2)
    EMaverage3 = ExpMovingAverage(df['Close'],MA3)

    #============================RSI================================
    rsi = rsiFunc(df['Close'])
    
    seuil70=np.repeat(70,len(df.index))
    seuil30=np.repeat(30,len(df.index))
    
    #===========================MACD================================
    emaslow,emafast,macdLine = computeMACD(df['Close']) #MACD line
    signalLine=ExpMovingAverage(macdLine,16)             #signal line
    hist = macdLine - signalLine                        #histogram
    
    #===========================MACD signals========================
    df['signal_MACD']=signauxMACD(signalLine,macdLine)
    #===========================EMA signals=========================
    df['signal_EMA']=signauxEMA(EMaverage1,EMaverage2,EMaverage3)
    #============================RSI signals========================
    df['signal_RSI']=signauxRSI(rsi)
    #===========================Strategy signals====================
    df['signal_strat']=shortLongStrategy(df['signal_EMA'],df['signal_RSI'],df['signal_MACD'])
    df['ordres']=buffer(df['signal_strat'])
    df['contrat']=contrat(df['ordres'])
    
    df['solde']=np.multiply(df['contrat'],df['Close'])
    solde_plot=abs(df['solde']).replace(0,np.nan) # solde_plot est la variable de plot
    
    df['returns']=investStrategy(df)
    
    #==============================PLOT PART===================================
    #=========================Plot Bougies + Volume============================
    ax_candles = go.Candlestick(x=df.index[-startingPoint:],
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
                             
     #=========================Plot RSI========================================                       
    axRSI=go.Scatter(x=df.index[-startingPoint:],
                    y=rsi[-startingPoint:],
                    xaxis='x2',
                    yaxis='y2',
                    name='RSI')
    
    axRSI70=go.Scatter(x=df.index[-startingPoint:],
                      y=seuil70[-startingPoint:],
                      xaxis='x2',
                      yaxis='y2',
                      mode='lines',
                      name='Seuil 70%',
                      showlegend=False,
                      hoverinfo='none')
    
    axRSI30=go.Scatter(x=df.index[-startingPoint:],
                      y=seuil30[-startingPoint:],
                      xaxis='x2',
                      yaxis='y2',
                      mode='lines',
                      name='Seuil 30%',
                      showlegend=False,
                      hoverinfo='none')
    
    #==========================Plot EMA's======================================
    ema1 = go.Scatter(x=df.index[-startingPoint:],
                     y=EMaverage1[-startingPoint:],
                     name='EMA'+str(MA1))
    ema2 = go.Scatter(x=df.index[-startingPoint:],
                     y=EMaverage2[-startingPoint:],
                     name='EMA'+str(MA2))
    ema3 = go.Scatter(x=df.index[-startingPoint:],
                     y=EMaverage3[-startingPoint:],
                     name='EMA'+str(MA3))
    
    #=========================Plot MACD======================================== 
    axMACDline=go.Scatter(x=df.index[-startingPoint:],
                          y=macdLine[-startingPoint:],
                          xaxis='x3',
                          yaxis='y3',
                          showlegend=False,
                          name='MACD Line')
    axSignalLine=go.Scatter(x=df.index[-startingPoint:],
                          y=signalLine[-startingPoint:],
                          xaxis='x3',
                          yaxis='y3',
                          showlegend=False,
                          name='Signal Line')
    
    couleur=[]
    for i in range(len(hist)):
        if hist[i]>=hist[i-1]:
            couleur.append('#00ff00')
        elif hist[i]<hist[i-1]:
            couleur.append('#ff0000')
    df['couleur']=couleur
    
    axHistBar=go.Bar(x=df.index[-startingPoint:],
                    y=hist[-startingPoint:],
                    xaxis='x3',
                    yaxis='y4',
                    name='MACD Variation',
                    marker=dict(color=df['couleur'][-startingPoint:]),
                    opacity=0.3)
    
    #=========================Plot Ordres trading=============================
    ax_ordres = go.Scatter(x=df.index[-startingPoint:],
                           y=solde_plot[-startingPoint:],
                           xaxis='x',
                           yaxis='y',
                           mode='markers',
                           marker=dict(color='#000080'),
                           name='ordres achat/vente')
    
    axTradeActions = go.Scatter(x=df.index[-startingPoint:],
                                y=df['returns'].cumsum()[-startingPoint:]*100,
                                xaxis='x6',
                                yaxis='y6',
                                mode='lines',
                                name='Trade strategy')
    
    axBenchDOW = go.Scatter(x=df.index[-startingPoint:],
                            y=benchDOW[-startingPoint:]*100,
                            xaxis='x6',
                            yaxis='y6',
                            mode='lines',
                            name='DOW JONES performance')
    
    axBenchNASDQ=go.Scatter(x=df.index[-startingPoint:],
                            y=benchNASDQ[-startingPoint:]*100,
                            xaxis='x6',
                            yaxis='y6',
                            mode='lines',
                            name='NASDAQ performance')
    
    #=========================Layout for plot=================================
    layout=go.Layout(plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    legend=dict(
                            font=dict(color='#FFF')),
                    xaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            linecolor='#5998ff',
                            linewidth=1,
                            mirror='ticks',
                            type="category",  #permet de supprimer les vides entre 22h et 15h
                            rangeslider=dict(
                                            visible=False)),
                    xaxis2=dict(
                                showgrid=False,
                                showticklabels=False,
                                linecolor='#5998ff',
                                linewidth=3,
                                mirror='ticks',
                                scaleanchor='x',
                                anchor='y2',
                                color='#5998ff',
                                type="category",
                                rangeslider=dict(
                                                visible=False)),
                    xaxis3=dict(
                                showticklabels=False,
                                scaleanchor='x',
                                anchor='y3',
                                linecolor='#5998ff',
                                linewidth=1,
                                mirror='ticks',
                                type="category",
                                rangeslider=dict(
                                                visible=False)),
                    xaxis4=dict(
                                showgrid=False,
                                showticklabels=False,
                                scaleanchor='x',
                                anchor='y4',
                                overlaying='x3',
                                linecolor='#5998ff',
                                linewidth=1,
                                mirror='ticks',
                                type="category",
                                rangeslider=dict(
                                                visible=False)),
                    xaxis6=dict(
                                showgrid=False,
                                showticklabels=True,
                                nticks=20,
                                scaleanchor='x',
                                anchor='y6',
                                linecolor='#5998ff',
                                linewidth=3,
                                mirror='ticks',
                                type="category",
                                rangeslider=dict(
                                                visible=False),
                                tickfont=dict(
                                            family='Old Standard TT, serif',
                                            size=9,
                                            color='white')),
                    yaxis=dict( # Main axis
                            showgrid=False,
                            showticklabels=True,
                            nticks=5,
                            linecolor='#5998ff',
                            linewidth=2,
                            mirror='ticks',
                            domain=[0.3,0.8],
                            anchor='free',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=12,
                                color='white'),
                            title='{} prices + indicators'.format(stock),
                            color='#5998ff'),
                    yaxis2=dict( # RSI index
                            showticklabels=True,
                            linecolor='#5998ff',
                            linewidth=3,
                            mirror='ticks',
                            domain=[0.8,1],
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=14,
                                color='white'),
                            title='RSI',
                            side='left',
                            color='#5998ff'),
                    yaxis3=dict( # MACD index
                            showgrid=False,
                            zeroline=False,
                            showticklabels=True,
                            color='#5998ff',
                            linewidth=3,
                            mirror=True,
                            side='left',
                            domain=[0.1,0.3],
                            anchor='x3',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=12,
                                color='white')),
                    yaxis4=dict( # MACD diff histogram
                            zeroline=False,
                            showgrid=False,
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=3,
                            mirror=True,
                            domain=[0.1,0.3],
                            title='MACD',
                            side='left',
                            overlaying='y3',
                            anchor='x4',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=14,
                                color='black'),
                            ),
                    yaxis5=dict( # volume
                            showgrid=False,
                            showticklabels=False,
                            color='#5998ff',
                            linewidth=3,
                            mirror='ticks',
                            domain=[0.3,0.8],
                            side='right',
                            overlaying='y',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=1,
                                color='black')),
                    yaxis6=dict( #performances stock + index
                            nticks=3,
                            zeroline=False,
                            showgrid=False,
                            linewidth=3,
                            linecolor='#5998ff',
                            mirror='ticks',
                            anchor='x6',
                            domain=[0,0.1],
                            color='white',
                            title='performance(%)',
                            side='right')
                    )
                           
    data=[ax_candles,ema1,ema2,ema3,axRSI,axRSI70,axRSI30,axMACDline,axSignalLine,axHistBar,axV,axTradeActions,axBenchDOW,axBenchNASDQ]
    fig = go.Figure(data=data,layout=layout)
    fig['layout'].update(title=stock+' prices and backtested strategy',titlefont=dict(color='#FFF'))
    return fig


def graphPortfolio(tickers):
    dowJonesIndex = get_index_google_finance_intraday('.DJI')
    nasdaqIndex=get_index_google_finance_intraday('.IXIC')
    benchDOW=np.log(dowJonesIndex['Close']).diff().cumsum()
    benchNASDQ=np.log(nasdaqIndex['Close']).diff().cumsum()
    df=pd.DataFrame()
    
    #============Data Size specification============

    tailleMin=999999999999
    for ticker in tickers:
        prices = get_google_finance_intraday(ticker,60,1)
        taille=len(prices)
        if(taille<=tailleMin):
            tailleMin=taille
    #Length value error : certaines valeurs ne sont pas captes par le provider
    #on fait le choix de prendre la taille minimale parmis l'ensemble de la data
    #en faisant l'hypothese que les donnees non captes ne sont pas suffisamment importante
    #pour qu'on ai a imputer les donnees

    for x in tickers:
        label=x
        prices=get_google_finance_intraday(x,60,1)
        close=prices['Close'][-tailleMin:]
        df.loc[:,'{}'.format(label)]=pd.Series(close.values)
    df.set_index(prices.index[-tailleMin:],inplace=True)
    startingPoint=len(dowJonesIndex['Close'][13-1:])
    
    #============================(à modifier) affectation des poids aleatoires=========================
    weights = np.linspace(0.1,1,len(tickers))
    weights /= weights.sum()
    weight_matrix=np.repeat([weights],len(close),axis=0)
    df_weight=pd.DataFrame(weight_matrix,columns=[tickers])

    #calculate the cumulative returns assuming our price distribution is log-normal
    returns = np.log(df).diff().cumsum()
    df_returns = pd.DataFrame(returns.values,index=df.index,columns=[tickers])
    
    for x in tickers:
        EMaverage1 = ExpMovingAverage(df[x],5)
        EMaverage2 = ExpMovingAverage(df[x],8)
        EMaverage3 = ExpMovingAverage(df[x],13)
        rsi = rsiFunc(df[x])
        emaslow,emafast,macdLine = computeMACD(df[x])
        signalLine=ExpMovingAverage(macdLine,9)
        signal_MACD=signauxMACD(signalLine,macdLine)
        signal_EMA=signauxEMA(EMaverage1,EMaverage2,EMaverage3)
        signal_RSI=signauxRSI(rsi)
        trade_strategy=shortLongStrategy(signal_EMA,signal_RSI,signal_MACD)
        df_returns['{} trade strategy'.format(x)]=buffer(trade_strategy)
        
    df_strat=pd.DataFrame()
    for x in tickers:
        df_strat['{} trade strategy'.format(x)]=np.multiply(df_returns['{} trade strategy'.format(x)].values, df_weight['{}'.format(x)].values)
    df_strat.set_index(df.index,inplace=True)
    
    #==============================Portfolio returns "global"===============================================
    strat_asset_log_returns=np.dot(df_strat,np.log(df).diff().transpose())
    portfolio_returns=np.diag(np.nan_to_num(strat_asset_log_returns)).sum()*100
    ptf_returns_record=np.diag(np.nan_to_num(strat_asset_log_returns))
    print("Portfolio's daily returns at {} : ".format(df.index[-1]),round(portfolio_returns,2),"%")
    
    portfolio_returns_plot=pd.Series(np.diag(np.nan_to_num(strat_asset_log_returns)).cumsum(),index=df.index)
    
    ax_ptf = go.Scatter(x=df.index[-startingPoint:],
                        y=round(portfolio_returns_plot[-startingPoint:]*100,2),
                        mode='lines',
                        name='Portfolio Returns')
    axBenchDOW = go.Scatter(x=df.index[-startingPoint:],
                            y=round(benchDOW[-startingPoint:]*100,2),
                            xaxis='x2',
                            yaxis='y2',
                            mode='lines',
                            name='DOW JONES performance')
    
    axBenchNASDQ=go.Scatter(x=df.index[-startingPoint:],
                            y=round(benchNASDQ[-startingPoint:]*100,2),
                            xaxis='x2',
                            yaxis='y2',
                            mode='lines',
                            name='NASDAQ performance')
    layout=go.Layout(plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    title='Portfolio performance VS Markets',
                    legend=dict(
                            font=dict(color='#FFF')),
                    titlefont=dict(color='#FFF'),
                    xaxis=dict(
                            nticks=20,
                            showgrid=False,
                            showticklabels=True,
                            type='category',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=12,
                                color='white')),
                    
                    xaxis2=dict(
                                scaleanchor='x',
                                showticklabels=False,
                                anchor='y2',
                                overlaying='x',
                                linecolor='#5998ff',
                                linewidth=3,
                                mirror='ticks',
                                type="category",
                                rangeslider=dict(
                                                visible=False)
                                ),
                    yaxis=dict(
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=12,
                                color='white')),
                    yaxis2=dict( #performances stock + index
                            nticks=3,
                            showticklabels=False,
                            zeroline=False,
                            showgrid=False,
                            linewidth=3,
                            linecolor='#5998ff',
                            mirror='ticks',
                            anchor='x2',
                            overlaying='y',
                            domain=[0,0.1],
                            color='white',
                            title='performance(%)',
                            side='left',
                            tickfont=dict(
                                family='Old Standard TT, serif',
                                size=12,
                                color='white'))
                    )
    data=[ax_ptf,axBenchDOW,axBenchNASDQ]
    fig=go.Figure(data=data,layout=layout)
    return fig,df,ptf_returns_record


def optimisationGraph(tickers): #Monte Carlo Simulation to graph optimal portfolio
    df=pd.DataFrame()
    
    #============Data Size specification============
    tailleMin=999999999999
    for ticker in tickers:
        prices = get_google_finance_intraday(ticker,60,1)
        taille=len(prices)
        if(taille<=tailleMin):
            tailleMin=taille

    for x in tickers:
        label=x
        prices=get_google_finance_intraday(x)
        close=prices['Close'][-tailleMin:]
        df.loc[:,'{}'.format(label)]=pd.Series(close.values)
    df.set_index(prices.index[-tailleMin:],inplace=True)
    
    #calculate the cumulative returns assuming our price distribution is log-normal
    returns = df.pct_change(1)
    df_returns = pd.DataFrame(returns.values,index=df.index,columns=[tickers])
    mean_daily_returns=df_returns.mean()
    cov_mat=df_returns.cov()
    
    num_random_portfolios=25000
    results=np.zeros((3,num_random_portfolios))
    allocation_scenarii=np.zeros((len(tickers),num_random_portfolios))
    
    for i in range(num_random_portfolios):
        weights=np.random.random(len(tickers))
        weights/=weights.sum()
        
        portfolio_return=np.sum(mean_daily_returns*weights)*252
        portfolio_std_dev=np.sqrt(np.dot(weights.T,np.dot(cov_mat,weights)))*np.sqrt(252)
        
        allocation_scenarii[:,i]=weights
        results[0,i]=portfolio_return
        results[1,i]=portfolio_std_dev
        results[2,i]=results[0,i]/results[1,i] #Sharpe Ratio calculation
    
    alloc_actifs=pd.DataFrame(allocation_scenarii.transpose(),columns=[tickers],index=results[2,:].transpose())
    results_frame=pd.DataFrame(results.T,columns=['returns','stdev','sharpe'])

    ax_vol_ret=go.Scatter(x=results_frame['stdev'],
                          y=results_frame['returns'],
                          mode='markers',
                          marker=dict(
                                  color=results_frame['sharpe'].values,
                                  colorscale='Jet',
                                  showscale=True,
                                  colorbar=dict(
                                                tickfont=dict(
                                                              color='white'))),
                          name='Portfolio Returns / volatility')
                          
    layout = go.Layout(title='Portfolio optimisation graph - Sharpe Ratio',
                       titlefont=dict(color='#FFF'),
                       legend=dict(
                            font=dict(color='#FFF')),
                       plot_bgcolor='#000000',
                       paper_bgcolor='#000000',
                       xaxis=dict(title='Volatility',
                                  titlefont=dict(color='#5998ff'),
                                  mirror='ticks',
                                  linecolor='#5998ff',
                                  linewidth=3,
                                  showgrid=True,
                                  tickfont=dict(
                                                family='Old Standard TT, serif',
                                                size=12,
                                                color='white')
                                  ),
                       yaxis=dict(title='Returns',
                                  titlefont=dict(color='#5998ff'),
                                  mirror='ticks',
                                  linecolor='#5998ff',
                                  linewidth=3,
                                  showgrid=True,
                                  tickfont=dict(
                                                family='Old Standard TT, serif',
                                                size=12,
                                                color='white')
                                  )
                       )
    print('Portfolio\'s best Sharpe Ratio :',round(results_frame['sharpe'].max(),2))
    data=[ax_vol_ret]
    fig=go.Figure(data=data,layout=layout)

    return fig,alloc_actifs
    
def ptfOptimisationGraph(df): #Monte Carlo Simulation to graph optimal strategy portfolio
    
    #calculate the cumulative returns assuming our price distribution is log-normal
    returns = df.pct_change(1)
    df_returns = pd.DataFrame(returns.values,index=df.index,columns=[tickers])
    mean_daily_returns=df_returns.mean()
    cov_mat=df_returns.cov()
    
    num_random_portfolios=25000
    results=np.zeros((3,num_random_portfolios))
    allocation_scenarii=np.zeros((len(tickers),num_random_portfolios))
    
    for i in range(num_random_portfolios):
        weights=np.random.random(len(tickers))
        weights/=weights.sum()
        
        portfolio_return=np.sum(mean_daily_returns*weights)*252
        portfolio_std_dev=np.sqrt(np.dot(weights.T,np.dot(cov_mat,weights)))*np.sqrt(252)
        
        allocation_scenarii[:,i]=weights
        results[0,i]=portfolio_return
        results[1,i]=portfolio_std_dev
        results[2,i]=results[0,i]/results[1,i] #Sharpe Ratio calculation
    
    alloc_actifs=pd.DataFrame(allocation_scenarii.transpose(),columns=[tickers],index=results[2,:].transpose())
    results_frame=pd.DataFrame(results.T,columns=['returns','stdev','sharpe'])
    
    ax_vol_ret=go.Scatter(x=results_frame['stdev']*100,
                          y=results_frame['returns']*100,
                          mode='markers',
                          marker=dict(
                                  color=results_frame['sharpe'].values,
                                  colorscale='Jet',
                                  showscale=True,
                                  colorbar=dict(
                                                tickfont=dict(
                                                              color='white'))),
                          name='Portfolio Returns / volatility')
                          
    layout = go.Layout(title='Portfolio optimisation graph - Sharpe Ratio',
                       titlefont=dict(color='#FFF'),
                       legend=dict(
                            font=dict(color='#FFF')),
                       plot_bgcolor='#000000',
                       paper_bgcolor='#000000',
                       xaxis=dict(title='Volatility (%)',
                                  titlefont=dict(color='#5998ff'),
                                  mirror='ticks',
                                  linecolor='#5998ff',
                                  linewidth=3,
                                  showgrid=True,
                                  tickfont=dict(
                                                family='Old Standard TT, serif',
                                                size=12,
                                                color='white')
                                  ),
                       yaxis=dict(title='Returns (%)',
                                  titlefont=dict(color='#5998ff'),
                                  mirror='ticks',
                                  linecolor='#5998ff',
                                  linewidth=3,
                                  showgrid=True,
                                  tickfont=dict(
                                                family='Old Standard TT, serif',
                                                size=12,
                                                color='white')
                                  )
                       )
    print('Portfolio\'s best Sharpe Ratio :',round(results_frame['sharpe'].max(),2))
    data=[ax_vol_ret]
    fig=go.Figure(data=data,layout=layout)
    
    return fig,alloc_actifs,results_frame

# input data
#EPA for Euronext Paris, NASD for NASDAQ companies    

while(True):
    try:
        portfolio_size=int(input("How many assets in your portfolio :"))
        break
    except Exception as e:
        print ('please put a number for your portfolio size')

tickers=[]
for i in range(portfolio_size):
    ticker = input("Stock number {} to pull on NASDAQ :".format(i+1))
    tickers.append(ticker)

period = 60
days = 1

config = {'showLink': False,'modeBarButtonsToRemove': ['sendDataToCloud']} #retirer le lien qui redirige vers plotly
stock_returns=pd.DataFrame()

for stock in tickers:
    fig=graphStockData(stock,5,8,13)
    py.plot(fig,config=config)

fig_ptf,df,ptf_ret_df=graphPortfolio(tickers)

#====================VaR Calculation of the strategy portfolio=============================
arranged_ret=sorted(ptf_ret_df,key=float) #arranging the strategy portfolio's returns
total_count=len(arranged_ret)
VaR_95_index=round((0.05)*total_count)
VaR_99_index=round((0.01)*total_count)
#VaR_999_index=round((0.001)*total_count)
VaR_95=arranged_ret[VaR_95_index]
VaR_99=arranged_ret[VaR_99_index]
#VaR_999=arranged_ret[VaR_999_index]

#=====================================Expected Shortfall (CVaR) calculation================
CVaR_95 = (1/VaR_95_index)*(sum(arranged_ret[:VaR_95_index]))
CVaR_99 = (1/VaR_99_index)*(sum(arranged_ret[:VaR_99_index]))
#CVaR_999 = (1/VaR_999_index)*(sum(arranged_ret[:VaR_999_index]))

df_arranged_ret=pd.DataFrame(ptf_ret_df,columns=['values'])
data_var=[df_arranged_ret['values']]
labels=['Potential losses']

VaR_Seuil=np.repeat(VaR_95,2000)

fig_var=ff.create_distplot(data_var,labels,bin_size=0.00005,show_hist=True,show_rug=False)
fig_var['layout'].update(title='Distribution of potential loss/returns',
                         titlefont=dict(color='#FFF'),
                         legend=dict(
                                     font=dict(color='#FFF')),
                         plot_bgcolor='#000000',
                         paper_bgcolor='#000000',
                         xaxis=dict(title='Loss -- Returns (%)',
                                  titlefont=dict(color='#5998ff'),
                                  mirror='ticks',
                                  linecolor='#5998ff',
                                  linewidth=3,
                                  tickfont=dict(
                                                family='Old Standard TT, serif',
                                                color='white')
                                  ),
                         yaxis=dict(title='Number of minutes',
                                  titlefont=dict(color='#5998ff'),
                                  mirror='ticks',
                                  linecolor='#5998ff',
                                  linewidth=3,
                                  showticklabels=False,
                                  tickfont=dict(
                                                family='Old Standard TT, serif',
                                                color='white')
                                  ),
                         annotations=[
                                     dict(
                                          x=VaR_95,
                                          y=0,
                                          xref='x',
                                          yref='y',
                                          text='VaR 95',
                                          font=dict(color='#F00'),
                                          showarrow=True,
                                          arrowhead=7,
                                          ax=0,
                                          ay=-350,
                                          arrowcolor='#F00'
                                             ),
                                     dict(
                                          x=CVaR_95,
                                          y=0,
                                          xref='x',
                                          yref='y',
                                          text='Expected Shorfall',
                                          font=dict(color='#F00'),
                                          showarrow=True,
                                          arrowhead=7,
                                          ax=0,
                                          ay=-300,
                                          arrowcolor='#F00'
                                          )
                                 ]
                        )
                                  
cdf=np.cumsum(arranged_ret)
cdf_norm=cdf/sum(cdf)
arranged_ret=np.array(arranged_ret)
data_concat=[arranged_ret,cdf_norm.cumsum()]
ax_cdf=go.Scatter(x=data_concat[0]*100,
                  y=data_concat[1],
                  mode='lines',
                  name='Potential losses')
ES_calc=abs(CVaR_95*100)
VaR_calc=1-abs(VaR_95*100)

TVaR_calc = VaR_calc-ES_calc/0.95
TVaR = -(1-TVaR_calc)


ax_VaR_plot=go.Scatter(x=data_concat[0]*100,
                      y=np.repeat(VaR_calc,len(data_concat[0])),
                      mode='lines',
                      hoverinfo='none',
                      name='VaR',
                      fill='tonexty',
                      line=dict(
                              color='#f34c4c'))

ax_TVaR_plot=go.Scatter(x=data_concat[0]*100,
                        y=np.repeat(TVaR_calc,len(data_concat[0])),
                        mode='lines',
                        hoverinfo='none',
                        name='TVaR',
                        fill='none',
                        line=dict(
                                color='#00ee00'))


layout_cdf=go.Layout(title='Cumulative Distribution Function curves with VaR + ES',
                     titlefont=dict(color='#FFF'),
                     legend=dict(
                                 font=dict(color='#FFF')),
                     plot_bgcolor='#000000',
                     paper_bgcolor='#000000',
                     xaxis=dict(title='Cumulative returns (%)',
                                titlefont=dict(color='#5998ff'),
                                mirror='ticks',
                                linecolor='#5998ff',
                                linewidth=3,
                                tickfont=dict(
                                              family='Old Standard TT, serif',
                                              color='white')
                                  ),
                     yaxis=dict(title='Probability of potential loss',
                                titlefont=dict(color='#5998ff'),
                                mirror='ticks',
                                linecolor='#5998ff',
                                linewidth=3,
                                range=[0,1],
                                tickfont=dict(
                                              family='Old Standard TT, serif',
                                              color='white')
                                  ),
                         annotations=[
                                     dict(
                                          x=0,
                                          y=TVaR_calc+(VaR_calc-TVaR_calc)/2,
                                          xref='x',
                                          yref='y',
                                          text='Expected Shortfall:{}%'.format(round(-ES_calc,3)),
                                          font=dict(color='#F00'),
                                          showarrow=True,
                                          arrowhead=7,
                                          ax=-100,
                                          ay=150,
                                          arrowcolor='#F00'
                                             )
                                      ]
                     )

data_cdf=[ax_cdf,ax_TVaR_plot,ax_VaR_plot]
fig_cdf=go.Figure(data=data_cdf,layout=layout_cdf)

py.plot(fig_var,config=config,filename='VaR_ES.html')

py.plot(fig_cdf,config=config,filename='VaR_ES_CDF.html')

py.plot(fig_ptf,config=config)

fig_opti_Ptf,allocation_actifs,results=ptfOptimisationGraph(df)
results.set_index('sharpe',inplace=True)
print('Yearly returns with optimal portfolio :',round(results.loc[results.index.max(),'returns']*100,2),'%')

ptf_sharpe_max=allocation_actifs.index.max()
meilleur_alloc = allocation_actifs.loc[ptf_sharpe_max]
meilleur_alloc.index.name='Assets'
meilleur_alloc.rename(index='Optimum asset allocation',inplace=True)
print('Optimum allocation :\n',round(meilleur_alloc,2))
print('============Chances of loss===================')
print('VaR(95%)=',round(VaR_95*100,2),'% of loss')
print('VaR(99%)=',round(VaR_99*100,2),'% of loss')
#print('VaR(99.9%)=',VaR_999*100,'% of loss')
print('=========How much shall we lose===============')
print('ES(95%)=',round(CVaR_95*100,2),'% of average loss on worst 5% of our returns')
print('ES(99%)=',round(CVaR_99*100,2),'% of average loss on worst 1% of our returns')
#print('ES(99.9%)',CVaR_999,'% of average loss on worst case scenario')

py.plot(fig_opti_Ptf,config=config)