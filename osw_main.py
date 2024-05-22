import xarray as xr
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from osw_ml_utils import proc_data
from osw_plot import plt_act_pred, plt_feat_imp
import json
import os
import GPUtil
import sys
     
        
def main(train_config_file):
        
        if len(GPUtil.getAvailable()) > 0:
                print("at least one gpu is available. Training will use gpu")
        else:
                print("no gpu found. Training will use cpu")        
        
        with open(train_config_file, "r") as openfile:
                train_config = json.load(openfile)      
        
        src_data = train_config["src_data"]                             # directory to save plots
        saveResPth = train_config["saveResPth"]                         # directory to save plots
        trainPlotCaption = train_config["trainPlotCaption"]             # caption for train data plot
        testPlotCaption = train_config["testPlotCaption"]               # caption for test data plot
        modelSavPth = train_config["model_name"]                        # name to save trained ML model
        model_version = train_config["model_version"]                   # model version
        force_cpu = train_config["force_cpu"]                           # flag to force cpu usage even when gpu is available

        # data processing variables
        featImpotCaption = train_config["featImpotCaption"]             # caption for feature importance plot
        applyQualPass = train_config["applyQualPass"]                   # flag to apply data quality pass
        appplyQualIceMask = train_config["appplyQualIceMask"]           # flag to apply data quality ice mask
        latLonEnc = train_config["latLonEnc"]                           # flag to encode latitude and longitude

        # ML model variables
        error_obj = train_config["error_obj"]                           # training error objective
        learning_r = train_config["learning_r"]                         # learning rate
        max_depth = train_config["max_depth"]                           # tree max depth
        l1_reg = train_config["l1_reg"]                                 # l1 regularization
        l2_reg = train_config["l2_reg"]                                 # l2 regularization
        
        model_file_name = modelSavPth.split(".")[0] + "_" + model_version + "." + modelSavPth.split(".")[1]
        full_model_pth = os.path.join(saveResPth, model_file_name)
        
        if os.path.isfile(os.path.join(full_model_pth)):
                print(f"ML model with the name '{model_file_name}' already exists in the specified output directory './{saveResPth}'. Exiting program...")
                sys.exit(0)
        
        if not os.path.exists(saveResPth):
                os.mkdir(saveResPth)
                print(f"created the directory './{saveResPth}', where results are written")
        else:
                print(f"results will be written to the directory: './{saveResPth}'")
        
        print("training configuration successfully loaded... ")
        
        data = xr.open_zarr(src_data)
        # process data to retrieve input and output data
        dataInp_arr, dataOut_arr = proc_data(data, qual_passFlag=applyQualPass, ice_flag=appplyQualIceMask, lat_lon_enc=latLonEnc)   

        # split data into training and testing 
        X_train, X_test, y_train, y_test = train_test_split(dataInp_arr, dataOut_arr, test_size=0.2, random_state=44)
        
        print("training data successfully loaded and processed... ")

        if len(GPUtil.getAvailable()) > 0 and not force_cpu:
        # build and configure xgboost model
                model = XGBRegressor(tree_method = "hist", device= "cuda",
                        reg=error_obj,
                        eta=learning_r,
                        max_depth=max_depth,                                                                
                        reg_alpha=l1_reg,
                        reg_lambda=l2_reg,
                        verbosity=0,
                        seed=42,
                        silent=True)
        else:
                model = XGBRegressor(tree_method = "hist", device= "cpu",
                        reg=error_obj,
                        eta=learning_r,
                        max_depth=max_depth,                                                                
                        reg_alpha=l1_reg,
                        reg_lambda=l2_reg,
                        verbosity=0,
                        seed=42,
                        silent=True)
                

        print("model training in progress... ")
        model.fit(X_train, y_train)
        
        # save trained model
        model.save_model(full_model_pth)
        # print(f"Model training finished. Model saved here: {os.path.join(saveResPth, modelSavPth)}")
        print(f"Model training finished. Model saved here: {full_model_pth}")
        
        model.load_model(full_model_pth)
        print("trained ML model loaded")
        
        print("Model testing in progress... ")
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_mae = round(mean_absolute_error(y_train, train_pred), 4)
        test_mae = round(mean_absolute_error(y_test, test_pred), 4)

        train_rmse = round(root_mean_squared_error(y_train, train_pred), 4)
        test_rmse = round(root_mean_squared_error(y_test, test_pred), 4)

        print(f"Training MAE: {train_mae}")
        print(f"Testing MAE: {test_mae}")

        print("Training RMSE:", train_rmse)
        print("Testing RMSE:", test_rmse)

        train_r2 = round(r2_score(y_train.to_numpy(), train_pred), 4)
        print(f"train r2 score is: {train_r2}") 

        test_r2 = round(r2_score(y_test.to_numpy(), test_pred), 4)
        print(f"test r2 score is: {test_r2}") 

        # plot actual outputs with predicted outputs
        plt_act_pred(X_train, y_train, train_pred, model_version, save_plt_pth=saveResPth, plot_caption=trainPlotCaption,
                rmse= train_rmse, r2=train_r2, ylabel="wind_era5", xlabel="sigma0_at_sp")

        plt_act_pred(X_test, y_test, test_pred, model_version, save_plt_pth=saveResPth, plot_caption=testPlotCaption,
                rmse= test_rmse, r2=test_r2, ylabel="wind_era5", xlabel="sigma0_at_sp")

        # feature importance
        plt_feat_imp(model, X_train, featImpotCaption, model_version, save_plt_pth=saveResPth)  

        
if __name__ == "__main__":
        train_config_file  = "./osw_config.json"
        main(train_config_file)