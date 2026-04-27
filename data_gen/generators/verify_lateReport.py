import pandas as pd, json
claims = json.load(open("data/claims.json"))
df = pd.DataFrame(claims)
df["days_to_file"] = (pd.to_datetime(df["filed_date"]) - pd.to_datetime(df["incident_date"])).dt.days
df["late_reporting"] = (df["days_to_file"] > 3).astype(int)
print(df.groupby("is_fraud")["late_reporting"].mean())