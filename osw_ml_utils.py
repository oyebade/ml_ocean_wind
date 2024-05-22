import numpy as np
import pandas as pd
import xarray as xr


def clean_data(df: pd.DataFrame, z_thresh: int=2)-> pd.DataFrame:
    """clean data by eliminating records that are beyond a
    certain thresholds in relation to z scores

    Args:
        df (data frame): data frame to clean
        z_thresh (int, optional): z score threshold. Defaults to 2.

    Returns:
        _type_: _description_
    """       
    z_scores = (df - df.mean())/df.std()             
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < z_thresh).all(axis=1)
    df = df[filtered_entries]
    return df

def encode_geo_coordinates(data, lat_col, lon_col):
    """
    Encode latitude and longitude into Cartesian coordinates and sine-cosine values.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the latitude and longitude.
    lat_col (str): Column name for latitude.
    lon_col (str): Column name for longitude.
    
    Returns:
    pd.DataFrame: DataFrame with encoded geographic features.
    """
    # Constants for the radius of the Earth in kilometers
    R = 6371
    
    # Convert latitude and longitude from degrees to radians
    data['lat_rad'] = np.deg2rad(data[lat_col])
    data['lon_rad'] = np.deg2rad(data[lon_col])
    
    # Cartesian coordinates
    data['x'] = R * np.cos(data['lat_rad']) * np.cos(data['lon_rad'])
    data['y'] = R * np.cos(data['lat_rad']) * np.sin(data['lon_rad'])
    data['z'] = R * np.sin(data['lat_rad'])
    
    # Sine and cosine transformations
    data['sin_lat'] = np.sin(data['lat_rad'])
    data['cos_lat'] = np.cos(data['lat_rad'])
    data['sin_lon'] = np.sin(data['lon_rad'])
    data['cos_lon'] = np.cos(data['lon_rad'])
    
    # Drop the intermediate columns if they are not needed
    data.drop(['lat_rad', 'lon_rad'], axis=1, inplace=True)
    data.drop([lat_col, lon_col], axis=1, inplace=True)
    
    return data


def proc_data(data:xr.Dataset, qual_passFlag:bool=True, ice_flag:bool=True, lat_lon_enc:bool=True)-> tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        data (xr.ndarray): data zarr
        qual_passFlag (bool, optional): flag to apply data quality pass mask. Defaults to True.
        ice_flag (bool, optional): flag to apply data ice mask. Defaults to True.
        lat_lon_enc (bool, optional): flag to encode latitude and longitude. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: processed input and output
    """    
    qual_mask = data["quality_pass"].values == 1
    qual_mask = qual_mask.astype(bool)

    ice_mask = data["quality_ice_flag"].values == 0
    ice_mask = ice_mask.astype(bool)
    
    print(f"dataset size: {len(data['sigma0_at_sp'].values)}")

    if qual_passFlag and ice_flag:
        dataInp_arr = xr.where(qual_mask * ice_mask, data["sigma0_at_sp"].values, np.nan)
        tx_id = xr.where(qual_mask * ice_mask, data["tx_id"].values, np.nan)
        sp_lat = xr.where(qual_mask * ice_mask, data["sp_lat"].values, np.nan)
        sp_lon = xr.where(qual_mask * ice_mask, data["sp_lon"].values, np.nan)
        incd_ang = xr.where(qual_mask * ice_mask, data["sp_incidence_angle"].values, np.nan)
        dataOut_arr = xr.where(qual_mask * ice_mask, data["wind_era5"].values, np.nan)
    else:
        dataInp_arr = xr.where(qual_mask, data["sigma0_at_sp"].values, np.nan)
        tx_id = xr.where(qual_mask, data["tx_id"].values, np.nan)
        sp_lat = xr.where(qual_mask, data["sp_lat"].values, np.nan)
        sp_lon = xr.where(qual_mask, data["sp_lon"].values, np.nan)
        incd_ang = xr.where(qual_mask, data["sp_incidence_angle"].values, np.nan)
        dataOut_arr = xr.where(qual_mask, data["wind_era5"].values, np.nan)

    dataInp_arr = 10*np.log10(dataInp_arr)
    dataInp_arr = pd.DataFrame(dataInp_arr)
    tx_id = pd.DataFrame(tx_id)
    sp_lat = pd.DataFrame(sp_lat)
    sp_lon = pd.DataFrame(sp_lon)
    incd_ang = pd.DataFrame(incd_ang)
    dataOut_arr = pd.DataFrame(dataOut_arr)

    df_comb = pd.concat([dataOut_arr, dataInp_arr, sp_lat, 
                        sp_lon, incd_ang, tx_id,], axis=1)
    df_comb.columns =["wind_era5", "sigma0_at_sp", "sp_lat", "sp_lon",
                    "sp_incidence_angle", "tx_id",]
    df_comb = df_comb.dropna()

    # convert tx_id to categorical and then integer
    df_comb.tx_id = pd.Categorical(pd.factorize(df_comb.tx_id)[0] + 1)
    df_comb["tx_id"] = df_comb["tx_id"].astype(np.uint16)

    if lat_lon_enc:
        # convert lat and long for ML
        df_comb = encode_geo_coordinates(df_comb, "sp_lat", "sp_lon")
        # extract input data
        # dataInp_arr = df_comb.loc[:, ["sigma0_at_sp", "x", "y", "z", "sin_lat", "cos_lat", "sin_lon", "cos_lon",
        #             "sp_incidence_angle", "tx_id"]]
        dataInp_arr = df_comb.loc[:, ["sigma0_at_sp", "x", "y", "z", "sp_incidence_angle", "tx_id"]]
        
    else:
        # extract input data
        dataInp_arr = df_comb.loc[:, ["sigma0_at_sp", "sp_lat", "sp_lon",
                        "sp_incidence_angle", "tx_id"]]

    dataOut_arr = df_comb.loc[:, "wind_era5"]
    return dataInp_arr, dataOut_arr
