import streamlit as st

from sovia.app.infra.DatabaseConnector import hausumringe_speichern, hausumringe_laden
from sovia.app.utils.file_handling import zwischenspeichern, find_shape_file, temp_dateien_loeschen


def upload_file():
    file = st.session_state.hausumringe_upload
    if file is not None:
        with st.spinner("Hausumringe werden importiert..."):
            erstellte_dateien = zwischenspeichern(file)
            shape_file_path = find_shape_file(erstellte_dateien)
            hausumringe_speichern(shape_file_path)
            temp_dateien_loeschen(erstellte_dateien)
        st.success("Import erfolgreich")


def main():
    st.title("Hausumringe Upload")
    st.text("Hier können die Hausumringe hochgeladen werden.")
    st.file_uploader("Hausumringe hochladen", key="hausumringe_upload", on_change=upload_file, type=["zip"])
    hausumringe = hausumringe_laden()
    if len(hausumringe) != 0:
        st.text("Vorschau der importierten Daten (Limitiert auf 100)")
        st.dataframe(hausumringe)


main()
