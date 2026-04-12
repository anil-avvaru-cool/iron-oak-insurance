import xgboost as xgb
import json
model = xgb.XGBRegressor()
model.load_model("ai/models/risk_scoring/risk_model.json")
feature_names = model.get_booster().feature_names
print(type(feature_names))
pretty_json = json.dumps(feature_names, indent=4)
print(pretty_json)