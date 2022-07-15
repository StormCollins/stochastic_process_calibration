import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


def read_vol_surface(
        file_path: str,
        sheet_name: str,
        data_date: datetime.date): # -> pd.DataFrame:
    data: pd.DataFrame = pd.read_excel(file_path, sheet_name=sheet_name)
    column_titles: pd.Index = data.columns
    moneyness: list[float] = list()

    for title in column_titles:
        if isinstance(title, float):
            moneyness.append(title)

    end_dates = data['End Date']
    tenors = list()
    for end_date in end_dates:
        tenors.append((end_date.date() - data_date).days / 365.0)

    vols = data[moneyness]
    vols.insert(0, 'Tenors', value=tenors)
    return vols


def get_vol(vol_surface, moneyness, tenor) -> float:
    """
    Returns an interpolated vol from a vol surface for a given moneyness and tenor.

    :param vol_surface:
    :return:
    """
    # TODO: Bilinear interpolation.

file_path: str = r'C:\GitLab\stochastic_process_calibration_2022\gbm\vol-surface-data-2022-06-30.xlsx'
sheet_name: str = 'S&P500 Clean Vol Surface'

data = read_vol_surface(file_path, sheet_name, datetime.date(2022, 6, 30))

print(data)


#print(data['End Date'])



# #Create the volatility surface
# def plot3D(X, Y, Z):
#     fig = plt.figure()
#     ax = Axes3D(fig, azim=-29, elev=50)
#
#     ax.plot(X, Y, Z, 'o')
#
#     plt.xlabel("expiry")
#     plt.ylabel("strike")
#     plt.show()
#
#     return plot3D(X,Y,Z)