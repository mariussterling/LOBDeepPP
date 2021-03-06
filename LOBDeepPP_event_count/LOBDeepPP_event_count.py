#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:40:18 2020

@author: ms
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
path_save = '.'

# %%
x = pd.read_csv('LOB_events_count.csv', delimiter=' ', header=None)
x.dropna(axis=1, how='all', inplace=True)
header = ['lines', 'words', 'bytes', 'file', 'ol']
x.rename(columns={i: h for i, h in enumerate(header)}, inplace=True)
x['day'] = x.file.str.slice(start=13, stop=15).map(int)

events = x.groupby('ol').sum().loc[:, 'lines']
events_per_day = x.groupby(['day', 'ol']).sum().loc[:, 'lines']

# %% LOB_ol_events
ax = (events / 10 ** 6).plot(x=range(1, 201))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
plt.ylabel('LOB($L$) changing events (in Million)')
plt.xlabel(f'Order book level $L$')
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events.{type}', bbox_inches='tight',
                dpi=300)
plt.xlim([0, 25])
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_zoom.{type}', bbox_inches='tight',
                dpi=300)
plt.show()

# %% LOB_ol_events_per_day

for i in range(6, 11):
    ax = (events_per_day.loc[i] / 10 ** 3).plot(label = i)
ax.legend()
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('LOB($L$) changing events (in Thousand)')
plt.xlabel(f'Order book level $L$')
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_day.{type}',
                bbox_inches='tight', dpi=300)
plt.xlim([0, 25])
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_day_zoom.{type}',
                bbox_inches='tight', dpi=300)
plt.show()
# %% LOB_ol_events_per_second
ax = (events / 27000 / 5).plot()  # (events / events.iloc[-1] * 100).plot()
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
plt.ylabel('Mean LOB($L$) changing events per second')
plt.xlabel('Order book level $L$')
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_second.{type}',
                bbox_inches='tight', dpi=300)
plt.xlim([0, 25])
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_second_zoom.{type}',
                bbox_inches='tight', dpi=300)
plt.show()

# %% LOB_ol_events_per_second_per_day
for i in range(6, 11):
    ax = (events_per_day.loc[i] / 27000).plot(label = i)
ax.legend()
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
plt.ylabel('Mean LOB($L$) changing events per second')
plt.xlabel('Order book level $L$')
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_second_per_day.{type}',
                bbox_inches='tight', dpi=300)
plt.xlim([0, 25])
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_second__per_day_zoom.{type}',
                bbox_inches='tight', dpi=300)
plt.show()

# %% LOB_ol_events_per_ol
ax = pd.Series(np.diff(events) / 10 ** 3, index=events.index[:-1]).plot()
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('Order book level $L$ changing events (in Thousand)')
plt.xlabel('Order book level $L$')
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_ol.{type}',
                bbox_inches='tight', dpi=300)
plt.xlim([0, 25])
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_ol_zoom.{type}',
                bbox_inches='tight', dpi=300)
plt.show()

# %% LOB_ol_events_per_second_per_ol
ax = pd.Series(np.diff(events) / 27000 / 5, index=events.index[:-1]).plot()
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
plt.ylabel('Mean Order book level $L$ changing events per second')
plt.xlabel('Order book level $L$')
plt.xlim([0, 25])

for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_second_per_ol.{type}',
                bbox_inches='tight', dpi=300)
plt.xlim([0, 25])
for type in ['pdf', 'png']:
    plt.savefig(f'{path_save}/LOB_ol_events_per_second_per_ol_zoom.{type}',
                bbox_inches='tight', dpi=300)
plt.show()
