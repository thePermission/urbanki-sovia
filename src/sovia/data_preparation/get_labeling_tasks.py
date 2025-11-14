from sovia.data_preparation.utils import create_connection, wms_links_viewname, get_path_to_data, label_tablename


def get_labeling_tasks(name: str):
    with create_connection() as con:
        sql = f"""
            WITH choosen_tasks AS (
            SELECT oi, ABS(klassifizierung - 0.5) as ref
            FROM {name}_klassifizierung as k
            )
            SELECT l.*, ref
            FROM choosen_tasks as c
            LEFT JOIN {wms_links_viewname} as l ON l.oi = c.oi
            LEFT JOIN {label_tablename} as lt on lt.oi = c.oi
            WHERE lt.oi IS NULL
            ORDER BY ref ASC
            LIMIT 1000
        """
        return con.sql(sql).to_csv(str(get_path_to_data(__file__) / "input/label_tasks" / f"{name}_tasks.csv"))


if __name__ == "__main__":
    name = "first_training"
    get_labeling_tasks(name)
