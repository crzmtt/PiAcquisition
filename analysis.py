# This is to analyze files obtained in the acquisition process
import pandas as pd
from pathlib import Path
import streamlit as st
import os, fnmatch
import numpy as np
from dataprep.clean import clean_lat_long
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px

def analysis():
    # select data among available
    dates = list(set([datefromfn(f) for f in fnmatch.filter(os.listdir('Out'), '*.csv')]))
    # remove faulty dates from dates
    #dates.remove('20230109')
    if len(dates) > 0:
        data=st.sidebar.selectbox('Select date to analyze',dates)
    else:
        st.write('No date available for analysis')
        quit()
    nsat = st.sidebar.selectbox('Select minimum number of satellites', [0,2,3,4,5],index=4)
    #data='20230#109'
    # streamlit stuff
    token=''

    # open csv files
    pwd = os.getcwd()
    outdir = Path(pwd, 'Out')
    fgps = pd.read_csv(Path(outdir, 'gps'+data+'.csv'), sep=';')
    fcan = pd.read_csv(Path(outdir, 'pcan'+data+'.csv'), sep=';')
    fbt  = pd.read_csv(Path(outdir, 'bt'+data+'.csv'), sep=';')

    print(fgps)
    print(fcan)
    print(fbt)

    fcan.rename(columns={'Time':'TIME_sec'},inplace=True)
    fcan = fcan.dropna(subset=['TIME_sec'])
    fgps = fgps.dropna(subset=['TIME_sec'])
    #fgps = fgps[fgps['satnum']>4]

    # Merge asoft
    all = pd.merge_asof(fcan, fgps, on='TIME_sec',direction='nearest',tolerance=1)
    assert isinstance(all, object)

    #all['latitude'] = all['latitude'] / 100.
    #all['longitude'] = all['longitude'] / 100.

    # Start plotting in streamlit
    #allgit = all.dropna(subset=['latitude','longitude'])
    if nsat==0:
        allgitp = all.copy(deep=True)
        allgit = all[all['satnum']>=2].copy(deep=True)
        allgitp['seconds'] = allgitp['TIME_sec'] - allgitp['TIME_sec'].values[0]
    else:
        allgit = all[all['satnum']>=nsat].copy(deep=True)
    #allgit = all.copy(deep=True)
    allgit['SN'] = np.where(allgit['latitude'] > 0, 'N', 'S')
    allgit['EW'] = np.where(allgit['longitude'] > 0, 'E', 'W')
    allgit['lattemp'] = allgit['latitude'] / 100.
    allgit['lattemp'] = allgit['lattemp'].astype('str')
    allgit['lontemp'] = allgit['longitude'] / 100.
    allgit['lontemp'] = allgit['lontemp'].astype('str')
    if len(allgit) == 0:
        st.error('Dataset is not valid, no lines to plot. Choose another date')
        st.experimental_rerun()
    allgit['coord'] = allgit.apply(lambda x: writecoord(x['lontemp'],x['EW'],x['lattemp'],x['SN']), axis=1)
    allgit.drop(columns={'latitude','longitude'},inplace=True)
    #print(allgit.to_string())
    allgit = clean_lat_long(allgit, "coord", split=True)
    print(allgit.columns)
    #allgit = pd.merge(allgit,allgitt,on='coord')
    allgit['latitude'] = allgit['latitude'].astype('float')
    allgit['longitude'] = allgit['longitude'].astype('float')
    allgit['seconds'] = allgit['TIME_sec'] - allgit['TIME_sec'].values[0]
    print(allgit.to_string())
    st.title(data + ' acquisition')

    vmax = allgit['VVehicle'].max()

    #col1, col2 = st.columns(2)
    col1, col2 = st.tabs(['Time series','Map'])
    with col1:
        pvars0 = ['VVehicle']
        if nsat == 0:
            allgitv = allgitp
        else:
            allgitv = allgit
        pvars = st.multiselect('Select variables to plot',allgitv.columns,default=pvars0)
        fig = px.line(allgitv,
            x="seconds",
            y=pvars,
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        #fig,ax = plt.subplots()
        #ax.plot(allgit['seconds'],allgit['VVehicle'],label='Velocità')
        #ax.plot(allgit['seconds'],allgit['SOC'],label='SOC')
        #ax.set_xlabel('Tempo [s]')
        #ax.set_ylabel('Velocità [km/h], SOC [kWh]')
        #ax.legend()
        #st.pyplot(fig)
        ##st.line_chart(allgit,x='seconds',y='VVehicle')
        ##st.line_chart(allgit,x='seconds',y='SOC')
        ##st.map(allgit)
        #fig,ax = plt.subplots()
        #ax.plot(allgit['seconds'],allgit['Motor_kW?'],label='Motor kW')
        #ax.plot(allgit['seconds'],allgit['ACPower?'],label='AC')
        #ax.plot(allgit['seconds'],allgit['heaterPower?'],label='Heat')
        #ax.set_xlabel('Tempo [s]')
        #ax.set_ylabel('Consumo motore, AC e Heat [kW]')
        #ax.legend()
        #st.pyplot(fig)'''

    with col2:
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=42.04,
                longitude=12.30,
                zoom=15,
                pitch=30,
            ),
            tooltip={"text": "Time: {seconds}\nLat: {latitude} Lon: {longitude}\n"
                             "Velocity: {VVehicle}\nMotor: {Motor_kW?}\nAC: {ACPower?}\nHeat: {heaterPower?}\n"
                             "SOC: {SOC}"},
            layers=[
                pdk.Layer(
                    'ColumnLayer',
                    data=allgit,
                    get_position='[longitude, latitude]',
                    get_elevation='[VVehicle]',
                    radius=3,
                    elevation_scale=2,
                    elevation_range=[0, 500],
                    get_fill_color='[200, 30, 0, 50]',
                    auto_highlight=True,
                    pickable=True,
                    extruded=True,
                ),
                #pdk.Layer(
                #    'ScatterplotLayer',
                #    data=allgit,
                #    get_position='[longitude, latitude]',
                #    get_color='[200, 30, 0, 160]',
                #    get_radius=3,
                #    pickable=True,
                #    extruded=True,
                #),
            ],
        ))
    #fig,ax = plt.subplots()
    #ax.plot(allgit['seconds'],allgit['VVehicle'],label='Velocità can')
    #ax.plot(allgit['seconds'],allgit['Speed'],label='Velocità gps')
    #ax.set_xlabel('Tempo [s]')
    #ax.set_ylabel('Velocità [km/h]')
    #ax.legend()
    #st.pyplot(fig)

    print('st fatto')

    return

def writecoord(lon,ew,lat,sn):
    lo,lor = lon.split('.')
    la,lar = lat.split('.')
    #return lo + '°' + lor[0:2]+"'"+lor[2:4]+'.'+lor[4:6]+'"'+ew+','+la+'°'+lar[0:2]+"'"+lar[2:4]+'.'+lar[4:6]+'"'+sn
    return la + '°' + lar[0:2]+"."+lar[2:6]+"'"+sn+','+lo+'°'+lor[0:2]+"."+lor[2:6]+"'"+ew

def datefromfn(fn):
    print(fn)
    return fn.split('.')[0][-8:]


if __name__ == '__main__':
    analysis()