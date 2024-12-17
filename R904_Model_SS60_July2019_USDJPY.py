# this uses code for Deep Nexus version 1 (code R904) completed July 2019

import numpy as np

from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask

from flask_cors import CORS

import pandas as pd
from sklearn import metrics, preprocessing
# Oanda imports...(to be simplified later)
from oandapyV20 import API
import oandapyV20.endpoints.instruments as v20instruments
from Oanda_Token import token
from collections import OrderedDict

from waitress import serve

app = Flask(__name__)
CORS(app)
app.config['PROPAGATE_EXCEPTIONS'] = True

def smooth_L1_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

# this can be used as a metric. 1 is a perfect score; an exact match to the target
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def SuperSmooth(ssdata, period=60):
    '''
    SuperSmooth - proposed by John Ehler to get rid of undersample noise.
    '''
    # close = df[['close']].values.ravel()
    f = (1.414 * np.pi) / period
    a = np.exp(-f)
    c2 = 2 * a * np.cos(f)
    c3 = - a * a
    c1 = 1 - c2 - c3

    ss = np.zeros(ssdata.shape)
    for i in range(len(ssdata)):
        if i < 2:
            continue
        else:
            ss[i] = c1 * (ssdata[i] + ssdata[i - 1]) * 0.5 + c2 * ss[i - 1] + c3 * ss[i - 2]
    # df_ss = pd.DataFrame(data=ss, index=df.index, columns=['ss_{i}'.format(i=period)])
    return ss


def get_model():
    global model
    model = load_model('.hdf5',
                       custom_objects={'smooth_L1_loss': smooth_L1_loss, 'coeff_determination': coeff_determination})

    model._make_predict_function()
    print('Model loaded!')

# any incoming data preprocessing, do it here'


def DataFrameFactory(r, colmap=None, conv=None):
    def convrec(r, m):
        """convrec - convert OANDA candle record.

        return array of values, dynamically constructed, corresponding with config in mapping m.
        """
        v = []
        for keys in [x.split(":") for x in m.keys()]:
            _v = r.get(keys[0])
            for k in keys[1:]:
                _v = _v.get(k)
            v.append(_v)

        return v

    record_converter = convrec if conv is None else conv
    column_map_ohlcv = OrderedDict([
        ('time', 'Date'),
        ('mid:o', 'Open'),
        ('mid:h', 'High'),
        ('mid:l', 'Low'),
        ('mid:c', 'Close'),
        #('volume', 'Vol')
    ])
    cmap = column_map_ohlcv if colmap is None else colmap

    df = pd.DataFrame([list(record_converter(rec, cmap)) for rec in r.get('candles')])
    df.columns = list(cmap.values())
    # df.rename(columns=colmap, inplace=True)  # no need to use rename, cmap values are ordered
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
    del df['Date']
    df = df.apply(pd.to_numeric)  # OANDA returns string values: make all numeric
    return df



print('Loading Keras model...')
get_model()

@app.route('/usdjpy', methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':

        if __name__ == "__main__":
            api = API(access_token=token)
            params = {
                "count": 600,  # number of bars to download
                "granularity": "M1"  # timeframe
            }
            # instruments = ["USD_ZAR", "EUR_USD", "GBP_USD", "USD_JPY", "ZAR_JPY", "NZD_USD", "AUD_USD"]
            instruments = ["USD_JPY"]
            df = dict()

            for instr in instruments:
                try:
                    r = v20instruments.InstrumentsCandles(instrument=instr,
                                                          params=params)
                    api.request(r)
                except Exception as err:
                    print("Error: {}".format(err))
                    exit(2)
                else:
                   df.update({instr: DataFrameFactory(r.response)})

            for I in instruments:
                print(df[I].tail())

                ticker = (df['USD_JPY'].iloc[:])

                ssdata0 = ticker.Close
                ssdata = np.array(ssdata0)

                ticker['ss'] = SuperSmooth(ssdata, period=60)

                ticker['target'] = np.log2(ticker.ss / ticker.ss.shift(180)).fillna(0.0001)
                ticker['target1'] = np.log2(ticker.ss / ticker.ss.shift(90)).fillna(0.0001)
                ticker['target2'] = np.log2(ticker.ss / ticker.ss.shift(1)).fillna(0.0001)

                targetnp = np.array(ticker.target)
                target1np = np.array(ticker.target1)
                target2np = np.array(ticker.target2)

                # inital pre-processing complete

                data = np.column_stack((targetnp, target1np, target2np))

                Xdata0 = np.delete(data, np.s_[0:185], axis=0)  # cut off initial nan's due to preprocessing: ~60+180

                # early scaling, can possibly be moved to after train/test split.
                scaler = preprocessing.RobustScaler()
                Xdata = scaler.fit_transform(Xdata0)

                X = np.array(Xdata[:, :])  # selects the entire range of columns available

                dX = []

                n_pre = 180
                n_post = 15

                for i in range(len(X) - 179):  # - 12):
                    dX.append(X[i:i + 180])

                dataX = np.array(dX)

# ------------------------
                print('Printing dataX...')
                print(dataX)
                print('About to predict...')
                predict = model.predict(dataX, batch_size=1000)  # .tolist()
                print('Prediction complete.')

                # simplified reshape and no inverse scaling; remain scaled for trade logic
                inverse_p = np.array(predict).reshape((len(predict), -1))

                price_action = pd.DataFrame({
                                             'p01': inverse_p[:, 0], 'p02': inverse_p[:, 1], 'p03': inverse_p[:, 2],
                                             'p04': inverse_p[:, 3], 'p05': inverse_p[:, 4], 'p06': inverse_p[:, 5],
                                             'p07': inverse_p[:, 6], 'p08': inverse_p[:, 7], 'p09': inverse_p[:, 8],
                                             'p10': inverse_p[:, 9], 'p11': inverse_p[:, 10], 'p12': inverse_p[:, 11],

                                             'p13': inverse_p[:, 12], 'p14': inverse_p[:, 13], 'p15': inverse_p[:, 14]})
                price_action['p01s5'] = price_action.p01.shift(5)  # index/column 15 in MT4
                price_action['p15s5'] = price_action.p15.shift(5)  # index/column 16 in MT4

                result1 = price_action.iloc[-1, :]
                print(result1.shape)
                print(result1)
                result = result1.to_json()
                return jsonify(result)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
	    "please wait until server has fully started"))

# waitress command to serve flask apps
serve(app, listen='127.0.0.3:80')


