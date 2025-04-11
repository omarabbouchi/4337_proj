import requests
import pandas as pd
from time import sleep

# TODO: replace base url w/ prod version for final prod dashboard 
# API docs: https://webapps.ilo.org/ilostat-files/Documents/SDMX_User_Guide.pdf 

def build_ilostat_url(dataflow_id, ref_area="USA", freq="M", measure="NB", sex="SEX_T", eco_code="ECO_ISIC4_TOTAL", start="2019", end="2023"):
    key = f"{ref_area}.{freq}..{sex}.{eco_code}"
    return f"https://www.ilo.org/sdmx-test/rest/data/ILO,{dataflow_id}/{key}?startPeriod={start}&endPeriod={end}&format=jsondata&detail=dataonly"

def fetch_ilostat_data(dataflow_ids, eco_codes, sleep_time=1):
    all_data = []

    for df_id in dataflow_ids:
        for eco in eco_codes:
            url = build_ilostat_url(df_id, eco_code=eco)
            try:
                r = requests.get(url, headers={"Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"})
                if r.status_code != 200:
                    continue
                j = r.json()
                structure = j["data"]["structure"]["dimensions"]["series"]
                obs_dim = j["data"]["structure"]["dimensions"]["observation"][0]["values"]
                series = j["data"]["dataSets"][0]["series"]

                for key, series_data in series.items():
                    dims = dict(zip([d['id'] for d in structure], key.split(":")))
                    for time_idx, val in series_data["observations"].items():
                        all_data.append({
                            "dataflow": df_id,
                            "industry": eco,
                            "time": obs_dim[int(time_idx)]['id'],
                            "value": val[0],
                            **dims
                        })
                sleep(sleep_time)
            except Exception as e:
                print(f"Error with {url}: {e}")
                continue

    return pd.DataFrame(all_data)

# Example use
dataflow_ids = [
    "DF_EMP_TEMP_SEX_ECO_NB",
    "DF_UNE_TUNE_SEX_AGE_NB",
    "DF_EES_XTMP_SEX_RT"
]

eco_codes = [
    "ECO_ISIC4_A", "ECO_ISIC4_B", "ECO_ISIC4_C", "ECO_ISIC4_D",
    "ECO_ISIC4_E", "ECO_ISIC4_F", "ECO_ISIC4_G", "ECO_ISIC4_H",
    "ECO_ISIC4_I", "ECO_ISIC4_J", "ECO_ISIC4_K", "ECO_ISIC4_L",
    "ECO_ISIC4_M", "ECO_ISIC4_N", "ECO_ISIC4_O", "ECO_ISIC4_P",
    "ECO_ISIC4_Q", "ECO_ISIC4_R", "ECO_ISIC4_S", "ECO_ISIC4_T",
    "ECO_ISIC4_U", "ECO_ISIC4_TOTAL"
]

df = fetch_ilostat_data(dataflow_ids, eco_codes)
df.to_csv("ilostat_job_market_data.csv", index=False)
