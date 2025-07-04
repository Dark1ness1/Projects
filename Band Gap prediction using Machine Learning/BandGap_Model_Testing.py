import pandas as pd
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
    )
import numpy as np
import joblib

if __name__ == "__main__":

    # Loading the saved classification and regression model

    XGBclf = joblib.load('./XGBClassifier.joblib')
    XGBreg = joblib.load('./XGBRegressor.joblib')

    # --- Test on new formulas ---
    print("Testing custom formulas...")
    test_formulas = ['GaN','CdTe','LiF','TiO2','CuSbS2','ZnS',
                    'Cu2ZnSnS4','PbTe','GaAs','ZnO']
    Eg_actual = [3.2,1.6,14.2,3.42,1.38,3.91,1.6,0.19,1.52,3.44]
    df_test = pd.DataFrame({"formula":test_formulas, "Eg_actual":Eg_actual})

    # Instantiate the featurizer
    ep = ElementProperty.from_preset("magpie")
    df_test["Composition"] = df_test["formula"].apply(lambda x: Composition(x))
    df_test = ep.featurize_dataframe(df_test, col_id="Composition", ignore_errors=True)
    features = ep.feature_labels()
    df_test = df_test.dropna(subset=features)

    # Classify and predict
    print("Classifying and predicting band gaps...")
    df_test["label"] = XGBclf.predict(df_test[features])
    df_pred_nm = df_test[df_test["label"]==1].copy()
    df_pred_nm["Eg_pred"] = XGBreg.predict(df_pred_nm[features])

    # Metrics
    rmse_new = np.sqrt(mean_squared_error(df_pred_nm["Eg_actual"], df_pred_nm["Eg_pred"]))
    r2_new = r2_score(df_pred_nm["Eg_actual"], df_pred_nm["Eg_pred"]) 
    mad_new = mean_absolute_error(df_pred_nm["Eg_actual"], df_pred_nm["Eg_pred"])
    print(f"Custom Test RMSE: {rmse_new:.3f}, R2: {r2_new:.3f}, MAD: {mad_new:.3f}")

    df_pred_nm["Deviation_%"] = (df_pred_nm["Eg_pred"]-df_pred_nm["Eg_actual"])/df_pred_nm["Eg_actual"]*100
    df_pred_nm["Eg_pred_str"] = df_pred_nm.apply(
        lambda r: f"{r['Eg_pred']:.2f} ({r['Deviation_%']:+.0f}%)", axis=1)
    print(df_pred_nm[["formula","Eg_actual","Eg_pred_str"]].to_string(index=False))

