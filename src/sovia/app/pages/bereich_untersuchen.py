import folium as fl
import streamlit as st
from folium import FeatureGroup
from streamlit_folium import st_folium

from sovia.app.infra.DatabaseConnector import gebiete_laden, gebiete_auflisten

if st.session_state.get('polys') is None:
    st.session_state['polys'] = []
if st.session_state.get('center') is None:
    st.session_state.center = [51.59397531704631, 7.136821746826173]
if st.session_state.get('zoom') is None:
    st.session_state.zoom = 14

st.title("Gebiet untersuchen")

left_column, right_column = st.columns([0.8, 0.2])
with right_column:
    st.selectbox("Gebiete", gebiete_auflisten(), key="gebiet_to_discover")
    st.button("untersuchen")

m = fl.Map(location=st.session_state.center, zoom_start=st.session_state.zoom)
with left_column:
    fg = FeatureGroup()
    for gebiet in gebiete_laden():
        for polygon in gebiet[2]:
            opacity = 0.8 if gebiet[0] == st.session_state.gebiet_to_discover else 0.3
            poly = fl.Polygon(
                locations=polygon,
                color=gebiet[1],
                fill=True,
                fill_color=gebiet[1],
                fill_opacity=opacity,
                tooltip=gebiet[0],
            )
            fg.add_child(poly)
    map_state = st_folium(m, use_container_width=True,
                          key="folium_map", feature_group_to_add=fg)
