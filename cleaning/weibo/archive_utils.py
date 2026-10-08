"""Helpers for the zipped (7z) daily Weibo dumps."""

import os


def extract_7z_files(source_folder, target_folder):
    import py7zr

    os.makedirs(target_folder, exist_ok=True)
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".7z"):
            file_path = os.path.join(source_folder, file_name)
            with py7zr.SevenZipFile(file_path, mode="r") as archive:
                archive.extractall(path=target_folder)
                print(f"Extracted: {file_name}")


def extract_single_7z_file(file_path, target_folder):
    import py7zr

    os.makedirs(target_folder, exist_ok=True)
    try:
        if file_path.endswith(".7z"):
            with py7zr.SevenZipFile(file_path, mode="r") as archive:
                archive.extractall(path=target_folder)
                print(f"Extracted: {file_path}")
                return "success"
    except Exception:
        return None


peak_months = {
    "072": ["2020-02", "2020-03"],
    "075": ["2022-07"],
    "084": ["2022-02", "2022-03"],
    "116": ["2023-02"],
    "123": ["2021-03", "2021-08", "2021-09", "2023-08"],
    "125": ["2020-02"],
    "167": ["2023-10"],
    "174": ["2023-03", "2023-04"],
    "176": ["2022-05", "2022-06", "2023-04", "2023-06"],
    "179": ["2022-06", "2022-08"],
}
