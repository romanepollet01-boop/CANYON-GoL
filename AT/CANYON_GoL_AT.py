import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import xarray as xr

base_dir = 'where_is_the_folder/CANYON-GoL/AT/'

## open the dataset
df = ... 

df_coast = pd.read_csv(base_dir+'../coastline_points_gol.csv',sep=',')
bathy = xr.open_dataset(base_dir + "../gebco_2025_n44.0_s41.0_w2.5_e7.0.nc")

def calculate_coast_dist(longitude, latitude):
    # compute the minimum distance (in km) between each observation point and the nearest coastline point
    # using Haversine formula
    #
    # parameters:
    # - longitude: array-like, longitude (in °)
    # - latitude: array-like, latitude (in °)
    #
    # return: 
    # distance: array-like, minimum distance to the coastline for each observation point (km)

    # convert coastline coordinates from degrees to radians
    coast_lat = np.radians(df_coast['lat'].values)
    coast_lon = np.radians(df_coast['lon'].values)

    # convert observation coordinates from degrees to radians
    df_lat = np.radians(latitude.values)
    df_lon = np.radians(longitude.values)

    # Earth's mean radius (km)
    R=6371.0
    # Array to store the minimum coastline distance for each point
    distance = np.empty(len(df_lat))

    # loop over all observation points
    for i in tqdm(range(len(df_lat))):
        # difference in latitude and longitude
        dlat = coast_lat - df_lat[i]
        dlon = coast_lon - df_lon[i]

        # Haversine formula
        a = (np.sin(dlat / 2)**2 + np.cos(df_lat[i]) * np.cos(coast_lat) * np.sin(dlon / 2)**2)
        c = 2 * np.arcsin(np.sqrt(a))

        # great-circle distance between the observation point and all coastline points
        dist = R * c

        # keep only the shortest distance
        distance[i] = np.min(dist)

    return distance


# model architecture : Multi-Layer Perceptron (MLP)
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_func):
        super(MLPModel, self).__init__() 

        # list used to store the successive layers of the network
        layers = []
        # number of input features for the first layer
        in_features = input_size 

        # build each hidden layer
        for h in hidden_layers:
            # linear transformation from the previous layer size to the current hidden size
            layers.append(nn.Linear(in_features, h)) 
            # activation function applied after the linear layer
            layers.append(activation_func) 
            in_features = h 

        # final output layer
        layers.append(nn.Linear(in_features, output_size))  
        self.model = nn.Sequential(*layers)

    # forward pass through the network
    def forward(self, x):
        return self.model(x) 

# custom scaled hyperbolic tangent activation function
class ScaledTanh(nn.Module):
    # scaling factor
    def __init__(self, A=4/3, alpha=1.7159):
        super().__init__()
        self.A = A
        self.alpha = alpha

    # comput the scaled tanh-like activation 
    def forward(self, x):
        return self.A * ((torch.exp(self.alpha * x) - 1) / (torch.exp(self.alpha * x) + 1))
    
    
def canyon_at(df, date, psal, pres, doxy, temp, co2_atm, lon, lat) :
    # Multi-Layer Perceptron to predict total alkalinity
    # Neural Network training by Romane Pollet from work by Marine Fourrier in LOV

    # input:
    # df : pandas.DataFrame
    # date: array-like, date (UTC) as string (format 'yyyy-mm-dd')
    # psal: array-like, practical salinity (PSU)
    # pres: array-like, pressure (dbar)
    # doxy: array-like, dissolved oxgyen (µmol/kg)
    # temp: array-like, temperature (°C)
    # co2_atm: array-like, atmospheric CO2 concentration (ppm)
    # lon: array-like, longitude (°E)
    # lat: array-like, latitude (°N)

    # output:
    # out: total alkalinity (at)
    #      - at_pred: predicted at
    #      - at_std: ensemble standard deviation

    date = pd.Series(pd.to_datetime(date))

    # calculate the input parameter required for the models
    df['coast_dist'] = calculate_coast_dist(lon, lat)
    df['pres_transf'] = pres / 2e4 + (1 / (1 + np.exp(-pres / 300))** 3)
    df['dec_year'] = date.dt.year + (date.dt.dayofyear - 1) / (365 + date.dt.is_leap_year.astype(int))
    df["bathy"] = - (bathy["elevation"].sel(lon=xr.DataArray(lon, dims="points"),lat=xr.DataArray(lat, dims="points"), method="nearest").values)

    # input combination
    features_model = ['pres_transf', 'psal', 'temp','doxy','dec_year','co2_atm','coast_dist','bathy']

    # load input and ouput scaler
    in_scaler = torch.load(base_dir+"input_scaler_at.pt", weights_only=False)
    out_scaler = torch.load(base_dir+"output_scaler_at.pt", weights_only=False)
    mean = pd.Series(in_scaler['mean'], index=features_model)
    std  = pd.Series(in_scaler['std'],  index=features_model)

    X_all = pd.DataFrame({
        'pres_transf' : df['pres_transf'].values, 'psal' : np.asarray(psal), 
        'temp' : np.asarray(temp), 'doxy' : np.asarray(doxy),
        'dec_year' : df['dec_year'].values, 'co2_atm' :np.asarray(co2_atm), 
        'coast_dist' : df['coast_dist'], 'bathy' : df["bathy"].values})[features_model]

    # normalisation of input parameters
    X_all_scaled = (2/3) * ((X_all - mean) / std)
    xtest_tensor = torch.tensor(X_all_scaled.values, dtype=torch.float32)

    # load neural network models and its architecture
    checkpoint = torch.load(base_dir+"cgol_at_models.pt", weights_only=False)
    models = [] 
    for model_info in checkpoint:
        m = MLPModel(
            input_size=model_info['input_size'],
            hidden_layers=[model_info['n1'], model_info['n2']],
            output_size=model_info['output_size'],
            activation_func=ScaledTanh())
        m.load_state_dict(model_info['model_state_dict'])
        m.eval()
        models.append(m)

    # prediction for each model
    predictions = []
    with torch.no_grad():
        for m in models : 
            pred = m(xtest_tensor).detach().cpu().numpy()
            predictions.append(pred)

    # denormalisation of output 
    predictions = np.stack(predictions, axis=0)
    predictions_real = (3/2) * predictions * out_scaler['std'] + out_scaler['mean']

    # mean and standard deviation of models
    y_pred = np.mean(predictions_real, axis=0)
    y_std = np.std(predictions_real, axis=0)

    # add predictions and uncertainties on dataframe
    df['at_pred'] = y_pred.squeeze()
    df['at_std'] = y_std.squeeze()

    return df  

# run alkalinity models on df 
# warning: change the name columns
df = canyon_at(df, df.date, df.psal, df.pres, df.doxy, df.temp, df.co2_atm, df.lon, df.lat)
