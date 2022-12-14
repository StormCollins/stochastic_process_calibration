# import pandas as pd
# import numpy as np
# from scipy.interpolate import interp1d
# from collections import namedtuple
#
# VolData = namedtuple('VolData', ['Tenors', 'VolSurface'])
#
# #
# # def read_vol_surface(excel_sheet_name: str) -> VolData:
# #     """
# #
# #     Reads in the Excel file containing the volatility surface.
# #
# #     :param excel_sheet_name: The name of the Excel file - atm-volatility-surface.xlsx'.
# #     :return: Returns the data read in from Excel.
# #     """
# #     data: pd.DataFrame = pd.read_excel(excel_sheet_name)
# #     tenors = data['Tenors'].tolist()
# #     volatilities = data['Quotes'].tolist()
# #     vol_data = VolData(tenors, volatilities)
# #     return vol_data
# #
# #
# # def get_vol(volatility_tenor: float, volatility_surface: VolData) -> float:
# #     """
# #     Returns an interpolated volatility from a volatility surface for a given tenor.
# #
# #     :param volatility_surface: The volatilities.
# #     :param volatility_tenor: The length of time remaining in the contract.
# #     :return: Interpolated volatility.
# #     """
# #
# #     bi_linear_interpolation: interp1d \
# #         = interp1d(volatility_surface.Tenors, volatility_surface.VolSurface, kind='linear')
# #     return bi_linear_interpolation(volatility_tenor)
#
# excel_file_path = '../gbm/atm-volatility-surface.xlsx'
# excel_records = pd.read_excel(excel_file_path)
# excel_records_df = excel_records.loc[:, ~excel_records.columns.str.contains('^Unnamed')]
# records_list_of_dict = excel_records_df.to_dict(orient='record')
# print(records_list_of_dict)
#
