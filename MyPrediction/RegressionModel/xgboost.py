import numpy as np
import pandas as pd

los_data = pd.read_csv('data/los_for_model.csv')
final_confirmed = los_data.drop(['stay_at_home_announced', 'stay_at_home_effective',
 'mean_temp', 'min_temp', 'max_temp', 'dewpoint', 'station_pressure',
 'visibility', 'wind_speed', 'max_wind_speed', 'fog' ,'rain', 'snow', 'hail',
 'thunder', 'tornado', 'mean_temp_3d_avg', 'mean_temp_5d_avg',
 'mean_temp_10d_avg' ,'mean_temp_15d_avg', 'max_temp_3d_avg',
 'max_temp_5d_avg', 'max_temp_10d_avg' ,'max_temp_15d_avg', 'min_temp_3d_avg',
 'min_temp_5d_avg' ,'min_temp_10d_avg', 'min_temp_15d_avg', 'dewpoint_3d_avg',
 'dewpoint_5d_avg', 'dewpoint_10d_avg', 'dewpoint_15d_avg'],axis=1)
print(final_confirmed)

final_confirmed.plot()

print('看到这条信息，即意味完成')