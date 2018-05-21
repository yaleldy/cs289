#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 23:12:54 2018

@author: YaleLang
"""

import numpy as np
import pandas as pd 
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap



## read the data
rawdata = pd.read_csv("us_census.csv")

ca = rawdata[rawdata.state == "CA"]
ca_lon = np.array(ca.longitude)
ca_lat = np.array(ca.latitude)
ca_loc = np.column_stack((ca_lat, ca_lon))

## kmeans

kmeans = cluster.KMeans(n_clusters = 6)
kmeans = kmeans.fit(ca_loc)
labels = kmeans.predict(ca_loc)
# create the map
m = Basemap(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-113,urcrnrlat=43,
        projection='lcc',lat_0 = 37.5,lon_0=-119)

m.readshapefile('CaliforniaCounty', name='california', drawbounds=True)

x,y = m(ca_lon, ca_lat)
plt.scatter(x,y,c = labels, s=1)
plt.savefig('kmeans.eps', format='eps', dpi=1000)


## SpectralClustering

sp = cluster.SpectralClustering(n_clusters = 6)
sp = sp.fit(ca_loc)
sp_labels = sp.fit_predict(ca_loc)

m = Basemap(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-113,urcrnrlat=43,
        projection='lcc',lat_0 = 37.5,lon_0=-119)

m.readshapefile('CaliforniaCounty', name='california', drawbounds=True)

x,y = m(ca_lon, ca_lat)
plt.scatter(x,y,c = sp_labels, s=1)
plt.savefig('sp.eps', format='eps', dpi=1000)


## AgglomerativeClustering

ac = cluster.AgglomerativeClustering(n_clusters = 6)
ac = ac.fit(ca_loc)
ac_labels = ac.fit_predict(ca_loc)


m = Basemap(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-113,urcrnrlat=43,
        projection='lcc',lat_0 = 37.5,lon_0=-119)

m.readshapefile('CaliforniaCounty', name='california', drawbounds=True)

x,y = m(ca_lon, ca_lat)
plt.scatter(x,y,c = ac_labels, s=1)
plt.savefig('ac.eps', format='eps', dpi=1000)

## DBSCAN
db = cluster.DBSCAN(min_samples = 40)
db = db.fit(ca_loc)
db_labels = db.fit_predict(ca_loc)

m = Basemap(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-113,urcrnrlat=43,
        projection='lcc',lat_0 = 37.5,lon_0=-119)

m.readshapefile('CaliforniaCounty', name='california', drawbounds=True)

x,y = m(ca_lon, ca_lat)
plt.scatter(x,y,c = db_labels, s=1)
plt.savefig('db.eps', format='eps', dpi=1000)

## HDBSCAN
import hdbscan
hdb = hdbscan.HDBSCAN(min_cluster_size = 40)
hdb = hdb.fit(ca_loc)
hdb_labels = hdb.fit_predict(ca_loc)

m = Basemap(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-113,urcrnrlat=43,
        projection='lcc',lat_0 = 37.5,lon_0=-119)

m.readshapefile('CaliforniaCounty', name='california', drawbounds=True)

x,y = m(ca_lon, ca_lat)
plt.scatter(x,y,c = hdb_labels, s=1)
plt.savefig('hdb.eps', format='eps', dpi=1000)



