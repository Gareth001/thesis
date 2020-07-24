from datetime import datetime

"""
List of useful util functions for preprocessing data
"""


def ensure_list(x):
    if not isinstance(x, list):
        return [x]
    return x


def standardise(dataframe, mapping):
    return dataframe.rename(columns=mapping)[list(mapping.values())]


def get_days(date0, date1):
    """
    Given redcap data in dates %Y-%m-%d
    convert to difference in days

    Parameters
    ----------
    date0: string
        redcaps dates in formate %Y-%m-%d
    date1: string
        redcaps dates in formate %Y-%m-%d
    """

    date0 = datetime.strptime(date0, "%Y-%m-%d")
    date1 = datetime.strptime(date1, "%Y-%m-%d")

    return (date1 - date0).days
