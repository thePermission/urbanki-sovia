from sovia.data_preparation.utils import create_connection, get_path_to_data, label_tablename, all_polygons_tablename, \
    wms_links_viewname, labeled_data_tablename


def load_labels(filename: str):
    with create_connection() as con:
        con.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {label_tablename} (
                OI VARCHAR NOT NULL,
                comparison_2020_2021 VARCHAR,
                comparison_2021_2022 VARCHAR,
                comparison_2022_2023 VARCHAR,
                comparison_2023_2024 VARCHAR,
                zustand2020 VARCHAR,
                zustand2021 VARCHAR,
                zustand2022 VARCHAR,
                zustand2023 VARCHAR,
                zustand2024 VARCHAR,
                PRIMARY KEY (OI)
            )
            """
        )
        con.sql(
            f"""
            INSERT OR REPLACE INTO {label_tablename}
                SELECT 
                    OI,
                    comparison_2020_2021,
                    comparison_2021_2022,
                    comparison_2022_2023,
                    comparison_2023_2024,
                    zustand2020,
                    zustand2021,
                    zustand2022,
                    zustand2023,
                    zustand2024 
                FROM read_csv('{get_path_to_data(__file__)}/input/labeled_data/{filename}')
            """)


def create_training_view():
    with create_connection() as con:
        con.sql(
            f"""
            CREATE OR REPLACE TABLE {labeled_data_tablename} AS (
            WITH
            label_as_number as (
                SELECT 
                    *,
                    CASE
                        WHEN comparison_2020_2021 = 'Dach wurde neugemacht' THEN 1
                        WHEN comparison_2021_2022 = 'Dach wurde neugemacht' THEN 2
                        WHEN comparison_2022_2023 = 'Dach wurde neugemacht' THEN 3
                        WHEN comparison_2023_2024 = 'Dach wurde neugemacht' THEN 4
                    ELSE 0 END as label_as_number
                from {label_tablename}
            ),
            data as (
                SELECT *,
                CASE WHEN label_as_number ==  1 AND zustand2021 IS NULL THEN 1 ELSE 0 END as label_2020_2021,
                CASE WHEN label_as_number BETWEEN 1 and 2 AND zustand2022 IS NULL THEN 1 ELSE 0 END as label_2020_2022,
                CASE WHEN label_as_number BETWEEN 1 and 3 AND zustand2023 IS NULL THEN 1 ELSE 0 END as label_2020_2023,
                CASE WHEN label_as_number BETWEEN 1 and 4 AND zustand2024 IS NULL THEN 1 ELSE 0 END as label_2020_2024,
                CASE WHEN label_as_number == 2 AND zustand2022 IS NULL THEN 1 ELSE 0 END as label_2021_2022,
                CASE WHEN label_as_number BETWEEN 2 and 3 AND zustand2023 IS NULL THEN 1 ELSE 0 END as label_2021_2023,
                CASE WHEN label_as_number BETWEEN 2 and 4 AND zustand2024 IS NULL THEN 1 ELSE 0 END as label_2021_2024,
                CASE WHEN label_as_number == 3 AND zustand2023 IS NULL THEN 1 ELSE 0 END as label_2022_2023,
                CASE WHEN label_as_number BETWEEN 3 and 4 AND zustand2024 IS NULL THEN 1 ELSE 0 END as label_2022_2024,
                CASE WHEN label_as_number == 4 AND zustand2024 IS NULL THEN 1 ELSE 0 END as label_2023_2024,
                COALESCE(zustand2021 == 'Solar auf dem Dach', false) as hat_solar_2021,
                COALESCE(zustand2022 == 'Solar auf dem Dach', false) as hat_solar_2022,
                COALESCE(zustand2023 == 'Solar auf dem Dach', false) as hat_solar_2023,
                COALESCE(zustand2024 == 'Solar auf dem Dach', false) as hat_solar_2024,
                COALESCE(zustand2021 != 'Dach nicht erkenntbar', true) as dach_erkennbar_2021,
                COALESCE(zustand2022 != 'Dach nicht erkenntbar', true) as dach_erkennbar_2022,
                COALESCE(zustand2023 != 'Dach nicht erkenntbar', true) as dach_erkennbar_2023,
                COALESCE(zustand2024 != 'Dach nicht erkenntbar', true) as dach_erkennbar_2024,
                COALESCE(zustand2021 == 'Haus im Bauprozess', false) as haus_im_bau_2021,
                COALESCE(zustand2022 == 'Haus im Bauprozess', false) as haus_im_bau_2022,
                COALESCE(zustand2023 == 'Haus im Bauprozess', false) as haus_im_bau_2023,
                COALESCE(zustand2024 == 'Haus im Bauprozess', false) as haus_im_bau_2024,
                COALESCE(comparison_2020_2021 == 'Dach wurde gereinigt', false) as dach_gereinigt_2021,
                COALESCE(comparison_2021_2022 == 'Dach wurde gereinigt', false) as dach_gereinigt_2022,
                COALESCE(comparison_2022_2023 == 'Dach wurde gereinigt', false) as dach_gereinigt_2023,
                COALESCE(comparison_2023_2024 == 'Dach wurde gereinigt', false) as dach_gereinigt_2024,
                FROM label_as_number l 
                LEFT JOIN {wms_links_viewname} as ap ON l.oi=ap.oi
            )
            SELECT
                oi as oi,
                2020 as year_1,
                2021 as year_2,
                label_2020_2021 as label,
                hat_solar_2021 as hat_solar,
                dach_erkennbar_2021 as dach_erkennbar,
                haus_im_bau_2021 as haus_im_bau,
                dach_gereinigt_2021 as dach_gereinigt,
                link_2020 as link_1,
                link_2021 as link_2
            FROM data UNION
            SELECT oi, 2020, 2022, label_2020_2022, hat_solar_2022, dach_erkennbar_2022, haus_im_bau_2022, dach_gereinigt_2022, link_2020, link_2022 FROM data UNION
            SELECT oi, 2020, 2023, label_2020_2023, hat_solar_2023, dach_erkennbar_2023, haus_im_bau_2023, dach_gereinigt_2023, link_2020, link_2023 FROM data UNION
            SELECT oi, 2020, 2024, label_2020_2024, hat_solar_2024, dach_erkennbar_2024, haus_im_bau_2024, dach_gereinigt_2024, link_2020, link_2024 FROM data UNION
            SELECT oi, 2021, 2022, label_2021_2022, hat_solar_2022, dach_erkennbar_2022, haus_im_bau_2022, dach_gereinigt_2022, link_2021, link_2022 FROM data UNION
            SELECT oi, 2021, 2023, label_2021_2023, hat_solar_2023, dach_erkennbar_2023, haus_im_bau_2023, dach_gereinigt_2023, link_2021, link_2023 FROM data UNION
            SELECT oi, 2021, 2024, label_2021_2024, hat_solar_2024, dach_erkennbar_2024, haus_im_bau_2024, dach_gereinigt_2024, link_2021, link_2024 FROM data UNION
            SELECT oi, 2022, 2023, label_2022_2023, hat_solar_2023, dach_erkennbar_2023, haus_im_bau_2023, dach_gereinigt_2023, link_2022, link_2023 FROM data UNION
            SELECT oi, 2022, 2024, label_2022_2024, hat_solar_2024, dach_erkennbar_2024, haus_im_bau_2024, dach_gereinigt_2024, link_2022, link_2024 FROM data UNION
            SELECT oi, 2023, 2024, label_2023_2024, hat_solar_2024, dach_erkennbar_2024, haus_im_bau_2024, dach_gereinigt_2024, link_2023, link_2024 FROM data
            )
            """
        )


def export_first_trainingsset():
    with create_connection() as con:
        con.sql(f"""
        WITH
        neues_dach as (SELECT * FROM {labeled_data_tablename} where label=1),
        dach_nicht_erkennbar as (SELECT * FROM {labeled_data_tablename} where label=0 AND not dach_erkennbar LIMIT 100),
        haus_im_bau as (SELECT * FROM {labeled_data_tablename} where label=0 AND haus_im_bau),
        hat_solar as (SELECT * FROM {labeled_data_tablename} WHERE label=0 AND hat_solar LIMIT 100),
        gereinigtes_dach as (SELECT * FROM {labeled_data_tablename} WHERE label=0 AND dach_gereinigt LIMIT 100),
        nicht_neu as (SELECT * FROM {labeled_data_tablename} WHERE label=0 AND NOT dach_gereinigt AND NOT hat_solar AND NOT haus_im_bau AND dach_erkennbar LIMIT 100)
        SELECT train_data.*, geom FROM (SELECT * FROM neues_dach UNION
        SELECT * FROM dach_nicht_erkennbar UNION 
        SELECT * FROM haus_im_bau UNION 
        SELECT * FROM hat_solar UNION
        SELECT * FROM gereinigtes_dach UNION
        SELECT * FROM nicht_neu) as train_data
        LEFT JOIN {all_polygons_tablename} as ap ON ap.oi = train_data.oi
        """).to_csv(file_name=f"{get_path_to_data(__file__)}/input/training_data/first_training.csv")


def export_second_trainingsset():
    with create_connection() as con:
        con.sql(f"""
        WITH
        klassifiziert as (SELECT OI, abs(klassifizierung - 0.5) as klasse FROM first_training_klassifizierung),
        trainingsset as (SELECT train.*, klasse FROM {labeled_data_tablename} as train LEFT JOIN klassifiziert as k ON train.oi=k.oi),
        neues_dach as (SELECT * FROM trainingsset where label=1 ORDER BY klasse),
        dach_nicht_erkennbar as (SELECT * FROM trainingsset where label=0 AND not dach_erkennbar ORDER BY klasse LIMIT 120),
        haus_im_bau as (SELECT * FROM trainingsset where label=0 AND haus_im_bau ORDER BY klasse LIMIT 120),
        hat_solar as (SELECT * FROM trainingsset WHERE label=0 AND hat_solar ORDER BY klasse LIMIT 120),
        gereinigtes_dach as (SELECT * FROM trainingsset WHERE label=0 AND dach_gereinigt ORDER BY klasse LIMIT 120),
        nicht_neu as (SELECT * FROM trainingsset WHERE label=0 AND NOT dach_gereinigt AND NOT hat_solar AND NOT haus_im_bau AND dach_erkennbar ORDER BY klasse LIMIT 120)
        SELECT train_data.*, geom, klasse FROM (SELECT * FROM neues_dach UNION
        SELECT * FROM dach_nicht_erkennbar UNION 
        SELECT * FROM haus_im_bau UNION 
        SELECT * FROM hat_solar UNION
        SELECT * FROM gereinigtes_dach UNION
        SELECT * FROM nicht_neu) as train_data
        LEFT JOIN {all_polygons_tablename} as ap ON ap.oi = train_data.oi
        """).to_csv(file_name=f"{get_path_to_data(__file__)}/input/training_data/second_training.csv")


def export_next_trainingsset():
    # die neuen Tasks plus die die man beim ersten training schon hatte
    with create_connection() as con:
        con.sql(f"""
        WITH trainings_oi AS (
            SELECT OI
            FROM read_csv('{get_path_to_data(__file__)}/input/labeled_data/second_labels.csv')
            )
        SELECT f_t.*, geom
        FROM trainings_oi as t 
        LEFT JOIN {labeled_data_tablename} as f_t ON f_t.oi = t.oi
        LEFT JOIN {all_polygons_tablename} as ap ON ap.oi = t.oi
        UNION
        SELECT * FROM read_csv('{get_path_to_data(__file__)}/input/training_data/first_training.csv')
        """).to_csv(file_name=f"{get_path_to_data(__file__)}/input/training_data/second_training.csv")


# load_labels("second_labels.csv")
# create_training_view()
# export_first_trainingsset()
# export_second_trainingsset()

#
# with create_connection() as con:
#     con.sql(
#         f"SELECT count(*) FROM {labeled_data_tablename} WHERE label=0 AND NOT dach_gereinigt AND NOT hat_solar AND NOT haus_im_bau AND dach_erkennbar").show()
