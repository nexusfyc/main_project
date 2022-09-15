import tweepy
import pandas as pd
import snscrape.modules.twitter as sntwitter
import datetime
import itertools

my_query = '(covid OR coronavirus) lang:en until:2020-03-16 since:2020-03-15 -filter:links'
df_city = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(my_query).get_items(), 5000))[['date', 'user', 'content']]

mid_array = []
key_word = 'Los Angeles'
key_word_lower = 'los angeles'

city_list = df_city.values.tolist()
for j in range(0,len(city_list)):
	if (city_list[j][1].get('location').find(key_word) != -1 or city_list[j][1].get('location').find(key_word_lower) != -1):
		mid_array.append(city_list[j])


df_ten_days = pd.DataFrame(mid_array,columns=['date','user','content'])

df_ten_days.to_csv('data/one_day.csv')