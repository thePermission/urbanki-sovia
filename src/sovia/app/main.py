import streamlit as st

from sovia.app.infra.DatabaseConnector import init

st.set_page_config(layout="wide", page_title="SoVIA")
init()
frontpage = st.Page("pages/Anleitung.py", title="Anleitung")
hausumringe_upload = st.Page("pages/hausumringe_upload.py", title="Hausumringe hochladen")
bereich_untersuchen = st.Page("pages/bereich_untersuchen.py", title="Gebiet untersuchen")
gebiete_bearbeiten = st.Page("pages/gebiete_bearbeiten.py", title="Gebiete bearbeiten")
pg = st.navigation([frontpage, hausumringe_upload, gebiete_bearbeiten, bereich_untersuchen])
pg.run()
