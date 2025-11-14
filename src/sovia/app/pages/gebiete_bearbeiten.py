import folium as fl
import streamlit as st
from folium import FeatureGroup
from folium.plugins import Draw
from streamlit_folium import st_folium

from sovia.app.infra.DatabaseConnector import gebiet_speichern, gebiete_auflisten, gebiet_loeschen, gebiete_laden

if st.session_state.get('poly') is None:
    st.session_state.poly = []
if st.session_state.get('center') is None:
    st.session_state.center = [51.59397531704631, 7.136821746826173]
if st.session_state.get('zoom') is None:
    st.session_state.zoom = 14

m = fl.Map(location=st.session_state.center, zoom_start=st.session_state.zoom)
draw_options = {
    "polyline": False,
    "marker": False,
    "circlemarker": False,
    "circle": False
}
Draw(draw_options=draw_options, ).add_to(m)
st.title("Gebiete bearbeiten")
left_column, right_column = st.columns([0.8, 0.2])
with left_column:
    fg = FeatureGroup()
    for gebiet in gebiete_laden():
        for polygon in gebiet[2]:
            poly = fl.Polygon(
                locations=polygon,
                color=gebiet[1],
                fill=True,
                fill_color=gebiet[1],
                fill_opacity=0.5,
                tooltip=gebiet[0],
            )
            fg.add_child(poly)
    map_state = st_folium(m, use_container_width=True,
                          key="folium_map", feature_group_to_add=fg)
with right_column:
    with st.container(border=True):
        st.subheader("Gebiete hinzufügen")
        st.text_input("Name", key="gebiet_to_save")
        if st.button("speichern"):
            if st.session_state.get("folium_map").get("all_drawings") is None:
                st.error("Keine Gebiete gezeichnet!")
            elif st.session_state.get("gebiet_to_save") is None or st.session_state.get("gebiet_to_save") == "":
                st.error("Kein Name angegeben!")
            else:
                gebiet_speichern(st.session_state.gebiet_to_save, st.session_state.folium_map["all_drawings"])
                st.session_state.center = [st.session_state.folium_map["center"]["lat"],
                                           st.session_state.folium_map["center"]["lng"]]
                st.session_state.zoom = st.session_state.folium_map["zoom"]
                st.rerun()
    with st.container(border=True):
        st.subheader("Gebiet löschen")
        st.selectbox("Gebiete", gebiete_auflisten(), key="gebiet_to_delete")


        def loeschen():
            gebiet_loeschen(st.session_state.gebiet_to_delete)


        st.button("löschen", on_click=loeschen)
