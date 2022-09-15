import tweepy
import pandas as pd
import snscrape.modules.twitter as sntwitter
import datetime
import itertools

def create_assist_date(datestart = None,dateend = None):
	# 创建日期辅助表

	if datestart is None:
		datestart = '2016-01-01'
	if dateend is None:
		dateend = datetime.datetime.now().strftime('%Y-%m-%d')

	# 转为日期格式
	datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
	dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
	date_list = []
	date_list.append(datestart.strftime('%Y-%m-%d'))
	while datestart<dateend:
		# 日期叠加一天
	    datestart+=datetime.timedelta(days=+1)
		# 日期转字符串存入列表
	    date_list.append(datestart.strftime('%Y-%m-%d'))
	return date_list

date_year = create_assist_date('2020-03-01', '2021-03-01')
my_query = '(covid OR coronavirus) lang:en until:{date_end} since:{date_begin} -filter:links'
key_word = 'Los Angeles'
key_word_lower = 'los angeles'

for i in range(0,len(date_year) - 1):
	mid_array = []
	count = 0
	new_query = my_query.format(date_end = date_year[i+1], date_begin = date_year[i])
	df_city = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(new_query).get_items(), 7000))[['date', 'user', 'content']]
	city_list = df_city.values.tolist()
	for j in range(0,len(city_list)):
		if (city_list[j][1].get('location').find(key_word) != -1 or city_list[j][1].get('location').find(key_word_lower) != -1):
			if (count == 50):
				break
			mid_array.append(city_list[j])
			count += 1
	df_ten_days = pd.DataFrame(mid_array,columns=['date','user','content'])
	final_df = df_ten_days.drop(['user'], axis=1)
	final_df.to_csv('data/los_angeles_tweets.csv', mode='a+')
# (covid OR coronavirus) (#losangeles) lang:en until:2020-03-16 since:2020-03-15 -filter:links


print("Accepted")