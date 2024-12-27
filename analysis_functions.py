import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf

def plot_functions():
    plt.plot(arr,color='#000000',label="Line")
    #plt.plot(avg, color='#0000ff',label="MVA")
    # plt.plot(arrmean, color='#0000Af',label="MEAN")
    # #plt.plot(avglin,color='#00ff00',label="LINAVG")
    #plt.plot(linplot,color='#ff0000',label="LIN")
    # #plt.plot(integr,color='#ffff00',label="INTEGRAL")
    #plt.plot(ema,color='#ff00ff',label="EMA")
    # plt.plot(diff,color='#00ffff',label="EMA_SMA_DIFF")
    #plt.plot(diffsign,color='#AA0000',label="EMA_SMA_DIFFERENTIAL") # first peak is max/min afterwardsgoes up/down
    plt.plot(higboil,color='#00AA00',label="Boiler HIGH")
    plt.plot(lowboil,color='#00AA00',label="Boiler LOW")

    impulses = plot_impulses(diffsignstd,arr.copy(),arr,lowboil,higboil,index=20,points=100,window=SENSITIVITY) # make stems in direction where line will go using ema sma differential
    plt.plot(impulses, color='#ff0000',label="impuls")
    plt.xticks(xlocs,xlabel,rotation="vertical")
    plt.grid(1)
    plt.legend()
    plt.show()

arr = np.random.rand(20)
def linear_regression(arr):
    x = range(arr.shape[0])
    linreg = stats.linregress(x,arr)
    linplot = linreg.intercept + linreg.slope * x
    return linplot

def ema_calc(curr,last,window):
    k = 2 / (window+1)
    ema = curr * k + last * 1-k
    return ema

def moving_average_with_linreg_startpoint(arr,window,number_of_linearregression_points): # moving average with deletion of every element that is influenced by the end of the data
    shape = arr.shape
    avgarr = np.zeros(arr.shape)

    arr = np.flip(arr)
    linreg = linear_regression(arr[-number_of_linearregression_points:-1])      
    arr = np.append(arr,linreg)

    for i in range(0,shape[0],1):
        avgarr[i] = arr[i:window+i].sum() / window
        
    avgarr = avgarr[:shape[0]]
    avgarr = np.flip(avgarr)
    return avgarr


def moving_average(arr,window): # moving average with deletion of every element that is influenced by the end of the data
    shape = arr.shape
    avgarr = np.zeros(arr.shape)

    arr = np.flip(arr)
    for i in range(window):
        arr = np.append(arr,arr[-1])

    for i in range(0,shape[0],1):
        avgarr[i] = arr[i:window+i].sum() / window
        
    avgarr = avgarr[:shape[0]]
    avgarr = np.flip(avgarr)
    return avgarr

# def moving_average_with_linear_regression(arr,window,number_of_linearregression_points,predict=False): # moving average with linear regression of the endpoint
                                        # seems unnecessary
#     shape = arr.shape
#     linreg = linear_regression(arr[-number_of_linearregression_points:-1])      
#     arr = np.append(arr,linreg)
#     avgarr = np.zeros(arr.shape)

#     for i in range(0,shape[0],1):
#         avg = arr[i:window+i].sum() / window
#         avgarr[i] = avg

#     avgarr = avgarr[:shape[0]]
#     return avgarr


def integral(arr,window,number_of_linearregression_points,predict=False): # integral instead of average with linear regression as endpoint

    shape = arr.shape
    avgarr = np.zeros(arr.shape)

    arr = np.flip(arr)

    if number_of_linearregression_points == None:
        for i in range(window):
            arr = np.append(arr,arr[-1])
    else:
        linreg = linear_regression(arr[-number_of_linearregression_points:-1])      
        arr = np.append(arr,linreg)

    
    for i in range(0,shape[0],1):
        avgarr[i] = np.trapz(arr[i:window+i]) / window
        
    avgarr = avgarr[:shape[0]]
    avgarr = np.flip(avgarr)
    return avgarr

def exponential_moving_average(arr,window):
    df = pd.DataFrame(arr)
    ema = df.ewm(span=window).mean()
    ema = ema.to_numpy()

    return ema

def emadiffsma(ema,sma,clip=False):   # if ema higher than sma then line is down instead of up (not going down but is already)
    ema = ema.squeeze()
    diff = np.subtract(ema,sma)      # going from -0.4 to 0 to positive is close to fall, 0.4 to 0 to negative is close to growth
    if clip:
        diff = np.clip(diff,-1.,1.)
    return diff

def emadiffsmaclassification(diff):     # for VERY SIMPLIFIED understanding negative number is BUY positive is SELL
    x = range(diff.shape[0])            # DO NOT USE LIKE THAT
                                        # first peak of amplitude is onset of growth/fall second peak is the peak of growth/fall = max Number, min Number/local maxima local minima
    dydx = np.zeros(diff.shape[0])
    singarr = np.sign(diff)
    signchange = ((np.roll(singarr, 1) - singarr) != 0).astype(int)
    indeces = np.where(signchange == 1)[0]

    for index in indeces:

        dy = np.diff(diff[index-1:index+1]) / np.diff(x[index-1:index+1])
        dydx[index-1:index+1] = dy

    return dydx

def rollingstd(arr,window):    
    avgarr = np.zeros(arr.shape)
    arr = np.flip(arr)

    for i in range(0,arr.shape[0],1):
        avgarr[i] = arr[i:window+i].std()
        

    avgarr = np.flip(avgarr)
    return avgarr

def boilingerband(arr,window):
    mva = moving_average(arr,window)
    stds = rollingstd(arr,window)

    highband = mva + 2 * stds
    lowband = mva - 2 * stds

    return highband, lowband

def plot_impulses(arr,plot_stems,line,lowboil,higboil,index,points,window):

    indeces_high = np.argwhere(arr>0)
    indeces_low = np.argwhere(arr<0)

    for i in indeces_high:
        cert = emasmadiffcertainty(diffsignstd,lowboil,higboil,index=i,points=100,window=SENSITIVITY,arr=arr)
        #print(cert)
        plt.stem(i,line[i]+cert,bottom=line[i])
        plot_stems[i] = line[i]+cert
    for i in indeces_low:
        cert = emasmadiffcertainty(diffsignstd,lowboil,higboil,index=i,points=100,window=SENSITIVITY,arr=arr)
        #print(cert)
        plt.stem(i,line[i]-cert,bottom=line[i])
        plot_stems[i] = line[i]-cert

    plot_stems = np.array(plot_stems)

    return plot_stems

def boiler_certainty_points(index,boilerlow,boilerhigh,points=100,):    # cut boiler into 100 points that are then used for ema sma diff prediction points. number of points giving the same result = certainty 

    boilerlow = np.append(boilerlow,boilerlow[-1])
    boilerhigh = np.append(boilerhigh,boilerhigh[-1])
    points = np.linspace(boilerlow[index],boilerhigh[index],num=points)

    return points


def emasmadiffcertainty(difforg,boilerlow,boilerhigh,index,points,window,arr):

    points = boiler_certainty_points(index,boilerlow,boilerhigh,points)
    results = []
    ind = [index-2,index-1,index+0,index+1,index+2,index+3,index+4,index+5,index+6]
    for p in points:
        #if index+6 > arr.size:
        while arr.size <= index+6:
            arr = np.append(arr,arr[-1])
        arr[index] = p

        sma = moving_average(arr,window)
        ema = exponential_moving_average(arr,window)

        diff = emadiffsma(ema,sma)
        diff_sign = emadiffsmaclassification(diff)
        diffi = diff_sign[ind]
        results.append(diffi)

    if index+6 >= difforg.size:
        while difforg.size <= index+6:
            difforg = np.append(difforg,difforg[-1])

    results = np.array(results)
    ressign = np.sign(results)
    diffi = difforg[ind]
    diffsign = np.sign(diffi)

    output = []
    for res in ressign:
        output.append(np.equal(diffsign,res))

    #print(output)

    output = np.array(output)
    out = output.sum()

    if out != 0:
        certainty = output.size / out
        #certainty = output.size / (output.size - out)
    else:
        certainty = 0

    #print(certainty)

    return certainty



CCOEY_data = yf.Ticker("UUUU")
CCOEY_hist = CCOEY_data.history("1mo")
label = CCOEY_hist.index.values
CCOEY_hist_meta = CCOEY_data.history_metadata

xlabel = np.datetime_as_string(label,unit="D")
xlabel = np.ravel([xlabel,xlabel],'F')

xlocs = np.array(range(xlabel.size))


#print(CCOEY_hist)

# get stock info

CCOEY_hist = CCOEY_hist.to_numpy()

open = CCOEY_hist[:,0]


close = CCOEY_hist[:,1]

arr = np.ravel([open,close],'F')
#arr = close

WINDOW = 20
SENSITIVITY = 8 # inverse
#avglin = moving_average_with_linear_regression(arr.copy(),WINDOW,WINDOW)

diffavg = moving_average(arr.copy(),SENSITIVITY)
avgstartlin = moving_average_with_linreg_startpoint(arr.copy(),WINDOW,WINDOW)
linplot = linear_regression(arr.copy())
integr = integral(arr.copy(),WINDOW,WINDOW)
diffema = exponential_moving_average(arr.copy(),SENSITIVITY)
diff = emadiffsma(diffema,diffavg)
diffsignstd = emadiffsmaclassification(diff)
diffsign = diffsignstd *4 + diffavg.mean()


#arr = arr - arr.mean()

ema = exponential_moving_average(arr.copy(),WINDOW)
avg = moving_average(arr.copy(),WINDOW)

rstd = rollingstd(arr.copy(),WINDOW)
higboil, lowboil = boilingerband(arr.copy(),WINDOW)
arrmean = np.full(arr.shape,arr.mean())

diff = diff + arr.mean()



plt.plot(arr,color='#000000',label="Line")
#plt.plot(avg, color='#0000ff',label="MVA")
# plt.plot(arrmean, color='#0000Af',label="MEAN")
# #plt.plot(avglin,color='#00ff00',label="LINAVG")
#plt.plot(linplot,color='#ff0000',label="LIN")
# #plt.plot(integr,color='#ffff00',label="INTEGRAL")
#plt.plot(ema,color='#ff00ff',label="EMA")
# plt.plot(diff,color='#00ffff',label="EMA_SMA_DIFF")
#plt.plot(diffsign,color='#AA0000',label="EMA_SMA_DIFFERENTIAL") # first peak is max/min afterwardsgoes up/down
plt.plot(higboil,color='#00AA00',label="Boiler HIGH")
plt.plot(lowboil,color='#00AA00',label="Boiler LOW")

impulses = plot_impulses(diffsignstd,arr.copy(),arr,lowboil,higboil,index=20,points=100,window=SENSITIVITY) # make stems in direction where line will go using ema sma differential
plt.plot(impulses, color='#ff0000',label="impuls")
plt.xticks(xlocs,xlabel,rotation="vertical")
plt.grid(1)
plt.legend()
plt.show()