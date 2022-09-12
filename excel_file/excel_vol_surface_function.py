import pandas as pd
from scipy.interpolate import interp1d
from collections import namedtuple

# data: pd.DataFrame = pd.read_excel('atm-volatility-surface.xlsx')
# print(data)

VolData = namedtuple('VolData', ['Tenors', 'VolSurface'])


def read_vol_surface(sheet_name: str) -> VolData:
    """

    Reads in the Excel file containing the volatility surface.

    :param sheet_name: The name of the Excel file - atm-volatility-surface.xlsx'.
    :return: Returns the data read in from Excel.
    """
    data: pd.DataFrame = pd.read_excel(sheet_name)
    tenors = data['Tenors'].tolist()
    volatilities = data['Quotes'].tolist()
    vol_data = VolData(tenors, volatilities)
    return vol_data


# print(read_vol_surface('atm-volatility-surface.xlsx'))

def get_vol(volatility_tenor: float, volatility_surface: VolData) -> float:
    """
    Returns an interpolated volatility from a volatility surface for a given tenor.

    :param volatility_surface: The volatilities.
    :param volatility_tenor: The length of time remaining in the contract.
    :return: Interpolated volatility.
    """

    bi_linear_interpolation: interp1d \
        = interp1d(volatility_surface.Tenors, volatility_surface.VolSurface, kind='linear')
    return bi_linear_interpolation(volatility_tenor)


# vol_surface: VolData = read_vol_surface('../excel_file/FEC_atm_vol_surface.xlsx')
# tenor: float = 0.25

# print(f'bi-linear interpolation is: {get_vol(tenor, vol_surface)}')

# xls = pd.ExcelFile('atm-volatility-surface.xlsx')
# df = xls.parse(xls.sheet_names[0], encoding='utf-16')
# print(df.to_dict())

# Converts Excel file into a dictionary.
# excel_file_path = '../excel_file/atm-volatility-surface.xlsx'
# excel_records = pd.read_excel(excel_file_path)
# excel_records_df = excel_records.loc[:, ~excel_records.columns.str.contains('^Unnamed')]
# records_list_of_dict=excel_records_df.to_dict(orient='record')
# print(records_list_of_dict)
