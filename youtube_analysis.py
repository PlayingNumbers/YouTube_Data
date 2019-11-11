# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:34:27 2019

@author: Ken
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
time_series = pd.read_csv('Totals_By_Date_With_More_Data.csv')
time_series_FB = pd.read_csv('Totals_By_Date_W_Facebook.csv')
videos = pd.read_csv('Video_Data_Expanded.csv')
video_timeline = pd.read_csv('Video by date.csv')

# dates to date time 

time_series['date_time'] = pd.to_datetime(time_series.date, infer_datetime_format = True)
time_series_FB['date_time'] = pd.to_datetime(time_series.date, infer_datetime_format = True)
video_timeline['date_time'] = pd.to_datetime(video_timeline.date, infer_datetime_format = True)
videos['Date_Published'] = pd.to_datetime(videos.Date_Published, infer_datetime_format = True)
fb_posts = time_series_FB[['date_time','Facebook_Intro_Post','Facebook_Video_post','FB_Likes']]

# video df editing 
videos['video_published'] = 1
videos_short = videos.loc[:,['video','Date_Published','video_title','video_published']]

ts_df = pd.merge(pd.merge(time_series, videos_short, left_on = 'date_time', right_on = 'Date_Published', how = 'left'), fb_posts, left_on = 'date_time',right_on='date_time', how = 'inner')

#descriptive statistics 

video_dates = list(videos_short[['Date_Published']].iloc[:,0])
recent_vids = list(filter(lambda x: x > pd.datetime(2018,8,15), video_dates))

#over time all stats 
# Show with fb post overlay as well

fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize = (15,8))
ax1.plot(ts_df.date_time,ts_df.views)
ax1.set_title('views over time')

ax2.plot(ts_df.date_time,ts_df.watch_time_minutes)
ax2.set_title('watch time (minutes) over time')

ax3.plot(ts_df.date_time, ts_df['average_percentage_viewed (%)'])
ax3.set_title('% of video viewed over time')

ax4.plot(ts_df.date_time, ts_df['video_thumbnail_impressions_ctr (%)'])
ax4.set_title('thumbnail impressions ctr over time')

ax5.plot(ts_df.date_time, ts_df.subscribers)
ax5.set_title('subscribers over time')

ax6.plot(ts_df.date_time, ts_df.likes)
ax6.set_title('likes over time')

for i in [ax1,ax2,ax3,ax4,ax5,ax6]:
    for xc in recent_vids:
        i.axvline(x=xc, color='red', linestyle='--', alpha = .3)
plt.tight_layout(pad=.5, w_pad=.5, h_pad=1.0)


# weekly rolling averages 
ts_df['views_weekly_avg'] = ts_df['views'].rolling(7).mean()
ts_df['watchtime_weekly_avg'] = ts_df['watch_time_minutes'].rolling(7).mean()
ts_df['pctview_weekly_avg'] = ts_df['average_percentage_viewed (%)'].rolling(7).mean()
ts_df['ctr_weekly_avg'] = ts_df['video_thumbnail_impressions_ctr (%)'].rolling(7).mean()
ts_df['subs_weekly_avg'] = ts_df['subscribers'].rolling(7).mean()
ts_df['likes_weekly_avg'] = ts_df['likes'].rolling(7).mean()

fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize = (15,8))
ax1.plot(ts_df.date_time,ts_df.views_weekly_avg)
ax1.set_title('views over time avg')

ax2.plot(ts_df.date_time,ts_df.watchtime_weekly_avg)
ax2.set_title('watch time (minutes) over time avg')

ax3.plot(ts_df.date_time, ts_df['pctview_weekly_avg'])
ax3.set_title('% of video viewed over time avg')

ax4.plot(ts_df.date_time, ts_df['ctr_weekly_avg'])
ax4.set_title('thumbnail impressions ctr over time avg')

ax5.plot(ts_df.date_time, ts_df.subs_weekly_avg)
ax5.set_title('subs over time avg')

ax6.plot(ts_df.date_time, ts_df.likes_weekly_avg)
ax6.set_title('likes over time avg')

for i in [ax1,ax2,ax3,ax4,ax5,ax6]:
    for xc in recent_vids:
        i.axvline(x=xc, color='red', linestyle='--', alpha = .3)
plt.tight_layout(pad=.5, w_pad=.5, h_pad=1.0)


# weekly rolling averages & Normal

fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize = (15,8))
ax1.plot(ts_df.date_time,ts_df.views, alpha = .5)
ax1.plot(ts_df.date_time,ts_df.views_weekly_avg, label = 'rolling_avg')
ax1.set_title('views over time avg')
ax1.legend()

ax2.plot(ts_df.date_time,ts_df.watch_time_minutes, alpha = .5)
ax2.plot(ts_df.date_time,ts_df.watchtime_weekly_avg, label = 'rolling_avg')
ax2.set_title('watch time (minutes) over time avg')
ax2.legend()

ax3.plot(ts_df.date_time, ts_df['average_percentage_viewed (%)'], alpha = .5)
ax3.plot(ts_df.date_time, ts_df['pctview_weekly_avg'], label = 'rolling_avg')
ax3.set_title('% of video viewed over time avg')
ax3.legend()

ax4.plot(ts_df.date_time, ts_df['video_thumbnail_impressions_ctr (%)'], alpha = .5)
ax4.plot(ts_df.date_time, ts_df['ctr_weekly_avg'], label = 'rolling_avg')
ax4.set_title('thumbnail impressions ctr over time avg')
ax4.legend()

ax5.plot(ts_df.date_time, ts_df.subscribers, alpha = .5)
ax5.plot(ts_df.date_time, ts_df.subs_weekly_avg, label = 'rolling_avg')
ax5.set_title('subs over time avg')
ax5.legend()

ax6.plot(ts_df.date_time, ts_df.likes, alpha = .5)
ax6.plot(ts_df.date_time, ts_df.likes_weekly_avg, label = 'rolling_avg')
ax6.set_title('likes over time avg')
ax6.legend()

for i in [ax1,ax2,ax3,ax4,ax5,ax6]:
    for xc in recent_vids:
        i.axvline(x=xc, color='red', linestyle='--', alpha = .2)
plt.tight_layout(pad=.5, w_pad=.5, h_pad=1.0)

plt.show()
#add lines for videos 
ts_df['day_of_week'] = ts_df.date_time.dt.dayofweek

dow = {0:'Monday',1:'Tuesday',2:'Wedensday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

ts_df['dow'] = ts_df.day_of_week.apply(lambda x: dow[x]) 

by_dow = pd.pivot_table(ts_df, values = ['watch_time_minutes','views','subscribers','likes'],columns='dow')
by_dow_sum = pd.pivot_table(ts_df, values = ['video_published','Facebook_Intro_Post','Facebook_Video_post'], columns='dow', aggfunc = 'sum')

vids_by_dow = ts_df[ts_df.video_published == 1 ]
vids_by_dow.dow.value_counts()
# views / watch time by day 

#remove days with FB posts and new videos 
no_interaction = ts_df[(ts_df.Facebook_Intro_Post != 1) & (ts_df.Facebook_Intro_Post != 2) & (ts_df.video_published != 1) & (ts_df.Facebook_Video_post != 1)]
by_dow_ni = pd.pivot_table(no_interaction, values = ['watch_time_minutes','views','subscribers','likes'],columns='dow')
by_dow_sum_ni = pd.pivot_table(no_interaction, values = ['video_published','Facebook_Intro_Post','Facebook_Video_post'], columns='dow', aggfunc = 'sum')

# views / watch time after new video 
ts_df['day_after_post'] =  ts_df.video_published.shift(1)
ts_df['day_after_fb_post'] = ts_df.Facebook_Intro_Post.shift(1)
ts_df['day_after_fb_vid_post'] = ts_df.Facebook_Video_post.shift(1)
ts_df['watchtime_previous'] = ts_df.watch_time_minutes.shift(1)

day_of_post = ts_df[(ts_df.Facebook_Intro_Post >= 1) | (ts_df.video_published >= 1) | (ts_df.Facebook_Video_post >= 1)]
day_of = pd.pivot_table(day_of_post, values = ['watch_time_minutes','views','subscribers','likes'], columns = 'dow')

day_after_post = ts_df[(ts_df.day_after_fb_post >= 1) | (ts_df.day_after_post >= 1) | (ts_df.day_after_fb_vid_post >= 1)]
day_after = pd.pivot_table(day_after_post, values = ['watch_time_minutes','views','subscribers','likes'], columns = 'dow')


#sort dow order 
fig2, [[ax7, ax8], [ax9, ax10]] = plt.subplots(2,2, figsize = (15,8))
ax7.plot(by_dow.columns, by_dow.loc['views'].values, label = 'All Days')
ax7.plot(by_dow_ni.columns, by_dow_ni.loc['views'].values, label = 'Days Without Posts')
ax7.plot(day_of.columns, day_of.loc['views'].values, label = 'Day of Posts')
ax7.plot(day_after.columns, day_after.loc['views'].values, label = 'Day After Posts')
ax7.legend()
ax7.set_title('views by day')

ax8.plot(by_dow.columns, by_dow.loc['watch_time_minutes'].values, label = 'All Days')
ax8.plot(by_dow_ni.columns, by_dow_ni.loc['watch_time_minutes'].values, label = 'Days Without Posts')
ax8.plot(day_of.columns, day_of.loc['watch_time_minutes'].values, label = 'Day of Posts')
ax8.plot(day_after.columns, day_after.loc['watch_time_minutes'].values, label = 'Day After Posts')
ax8.legend()
ax8.set_title('watch time by day')

ax9.plot(by_dow.columns, by_dow.loc['subscribers'].values, label = 'All Days')
ax9.plot(by_dow_ni.columns, by_dow_ni.loc['subscribers'].values, label = 'Days Without Posts')
ax9.plot(day_of.columns, day_of.loc['subscribers'].values, label = 'Day of Posts')
ax9.plot(day_after.columns, day_after.loc['subscribers'].values, label = 'Day After Posts')
ax9.legend()
ax9.set_title('subs by day')

ax10.plot(by_dow.columns, by_dow.loc['likes'].values, label = 'All Days')
ax10.plot(by_dow_ni.columns, by_dow_ni.loc['likes'].values, label = 'Days Without Posts')
ax10.plot(day_of.columns, day_of.loc['likes'].values, label = 'Day of Posts')
ax10.plot(day_after.columns, day_after.loc['likes'].values, label = 'Day After Posts')
ax10.legend()
ax10.set_title('likes by day')

plt.legend()
plt.show()

        

        
#performance of videos by day posted 
videos['day_of_week_num'] = videos.Date_Published.dt.dayofweek
videos['dow'] = videos.day_of_week_num.apply(lambda x: dow[x])
videos['days_published'] = videos.Date_Published.apply(lambda x: (pd.datetime(2019,10,17)-x).days)
videos['watchtime_day'] = videos.watch_time_minutes / videos.days_published
videos['views_day'] = videos.views / videos.days_published
videos['likes_day'] = videos.likes / videos.days_published
videos['subscribers_day'] = videos.subscribers / videos.days_published

pd.pivot_table(videos, columns = 'dow', values = 'watchtime_day',aggfunc = 'count')
pd.pivot_table(videos, columns = 'dow', values = ['watchtime_day','views_day','likes_day','subscribers_day'])
by_dow.loc['views'].values


by_dow.columns


#corr plot for watch time 
corrdata = ts_df[['watch_time_minutes','subscribers','watchtime_previous','video_published','Facebook_Intro_Post','Facebook_Video_post','day_after_post','day_after_fb_post','day_after_fb_vid_post']].fillna(0)

corr = corrdata.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

  
#corrplot for subscribers
corrdata = ts_df[['subscribers','watchtime_previous','video_published','Facebook_Intro_Post','Facebook_Video_post','day_after_post','day_after_fb_post','day_after_fb_vid_post']].fillna(0)

corr = corrdata.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
        
        
#length of title, length of video, punctuation in title, what is my best video? regression 
regression_data_X = pd.get_dummies(ts_df[['watchtime_previous','dow','video_published','Facebook_Intro_Post','Facebook_Video_post','day_after_post','day_after_fb_post','day_after_fb_vid_post']]).fillna(0)
regression_data_Y = ts_df[['watch_time_minutes']]
regression_data_Y_Subs = ts_df[['subscribers']]



from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(regression_data_X, regression_data_Y)
reg.score(regression_data_X,regression_data_Y)

import statsmodels.api as sm
regression_data_X = sm.add_constant(regression_data_X)
model_reg_watchtime = sm.OLS(regression_data_Y, regression_data_X).fit()
model_reg_watchtime.summary()

model_reg_subs = sm.OLS(regression_data_Y_Subs, regression_data_X).fit()
model_reg_subs.summary()

# create corplot



