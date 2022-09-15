import tweepy

consumer_key = 'UAdItglsqspGkoqkXtULQZkj2'
consumer_secret = 'zcTxzJMV2BcP4lnqG9DCX78YqOwVcme5L7NCxozkFWYzPZ93W7'
access_token_key = 'xxxxxxxxxxxxx'
access_token_secret = 'xxxxxxxxxxxxxxxx'

# tweepy credential and authorization settings
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)