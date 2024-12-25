# SOLAR RCA ALGORITHM
# Date: 11/04/2023

import pandas as pd
import numpy as np
import psycopg2 as pg
import itertools
import random
from datetime import datetime as dt
# from collections import defaultdict as dd
# import datetime
# import json

# Get data from database and return it in the matrix
def get_from_database():
    conn = pg.connect(host='localhost', database='solar_performacne_db', port='54321',
                      user='postgres', password='1235')
    curr = conn.cursor()
    curr.execute(""" SELECT siteid, date, solarkwh, solarkwhtarget,  actualloadkw
                     FROM performance;
                    """)
    # DATA STORED IN A LIST
    data = curr.fetchall()

    curr.close()
    conn.close()

    return data

# Group data on siteid basis and return results in matrix.
def sorting_data(data):
    # Grouping Data by Sites
    sort_data = sorted(data, key=lambda x: x[0])
    final_data = []

    for key, group in itertools.groupby(sort_data, key=lambda x: x[0]):
        final_data.append(list(group))

    return final_data

# Each Column is converted into a list. Return every column (i.e., parameter) list.
def data_list(col_no, single_data):
    data = [a[col_no] for a in single_data]

    #     To Convert None Vlaues to Zero
    #     data = [0 if val is None else val for val in data]
    return data


# Data can contain NULL values. Convert them to Zero
def none_checker(i, lastmonthdata):
    lastmonth = [rows[i] for rows in lastmonthdata]
    non_flag = False if None not in lastmonth else True
    lastmonth = [0 if elem is None else elem for elem in lastmonth]

    return lastmonth, non_flag


# Data Parsing function on basis of time.
def lastmonthdata_cal(a, one_site_data):
        lastmonthdata = []
        # row = []

        month = dt.now().month
        year = dt.now().year

        #     Date for the last Month
        for j in range(len(one_site_data[0])):
            if one_site_data[0][j].month == month - a and one_site_data[0][j].year == year:
                lastmonthdata.append([row[j] for row in one_site_data])

        date, _ = none_checker(0, lastmonthdata)

        # Here i=1 represents first col in last month that is solarkwh
        solarkwh_lastmonth, solarkwh_lastmonth_flag = none_checker(1, lastmonthdata)
        solartargetkwh_lastmonth, solartargetkwh_lastmonth_flag = none_checker(2, lastmonthdata)
        actualload_lastmonth, actualload_lastmonth_flag = none_checker(3, lastmonthdata)
        solarloss_lastmonth, solarloss_lastmonth_flag = none_checker(4, lastmonthdata)
        faultyAlarm_lastmonth, faultyAlarm_lastmonth_flag = none_checker(5, lastmonthdata)
        missingAlarms_lastmonth, missingAlarms_lastmonth_flag = none_checker(6, lastmonthdata)
        weather_lastmonth, weather_lastmonth_flag = none_checker(7, lastmonthdata)


        if len(solarkwh_lastmonth) == 0:
            max_solarkwh_lastmonth = 0
            avg_solarkwh_lastmonth = 0
            avg_solartargetkwh_lastmonth = 0
            avg_actualload_lastmonth = 0
            avg_solarloss_lastmonth = 0

        else:
            max_solarkwh_lastmonth = np.max(solarkwh_lastmonth)
            avg_solarkwh_lastmonth = np.mean(solarkwh_lastmonth)
            avg_solartargetkwh_lastmonth = np.mean(solartargetkwh_lastmonth)
            avg_actualload_lastmonth = np.mean(actualload_lastmonth)
            avg_solarloss_lastmonth = np.mean(solarloss_lastmonth)


        return date, solarkwh_lastmonth, solarkwh_lastmonth_flag, solartargetkwh_lastmonth, solartargetkwh_lastmonth_flag, actualload_lastmonth, actualload_lastmonth_flag, solarloss_lastmonth, solarloss_lastmonth_flag, faultyAlarm_lastmonth, faultyAlarm_lastmonth_flag, missingAlarms_lastmonth, missingAlarms_lastmonth_flag, weather_lastmonth, weather_lastmonth_flag, max_solarkwh_lastmonth, avg_solarkwh_lastmonth, avg_solartargetkwh_lastmonth, avg_actualload_lastmonth, avg_solarloss_lastmonth

# Algorithm 1st Part. Contains Field_Challenges and Planning
def solarPresent(date, solarkwh, avg_solarkw, avg_solarTarget, avg_actualLoad, avg_solarLoss, max_solarkWh_start,
                 max_solarkwh_month, faulty, missing):
    filed_challenges = []
    planning = []

    if avg_solarkw * (1 - avg_solarLoss) < avg_actualLoad:
        planning.append('Solar Under-Size1')
    if max_solarkWh_start < avg_solarTarget * 0.4:  # From Starts
        filed_challenges.append('Partial Panels Connected2')
    for i in faulty:
        if i == True:  # FAULTY ALARMS
            filed_challenges.append('Solar Faulty3')
            break
        else:  # ROUTINE S6
            daynum = 0
            faultycount = 0
            for j in range(1, len(date)):
                if max_solarkwh_month > 0.4 * avg_solarTarget:
                    if (solarkwh[j] < (solarkwh[j - 1] * 0.7)) and (
                            solarkwh[j] > (solarkwh[j - 1] * 0.5)):  # For Weather No routine is provided
                        faultcount = 1
                        daynum = i
                        break
            solar_till_dynum = np.mean([solarkwh[:daynum]])
            solar_till_end = np.mean([solarkwh[daynum:]])
            if faultycount == 1 and solar_till_dynum > solar_till_end * 0.7:
                filed_challenges.append('Solar Faulty4')

    for k in missing:
        if k == True:  # MISSING ALARMS
            filed_challenges.append('Solar Missing5')
        else:
            daynum = 0
            miscount = 0
            for l in range(1, len(date)):
                if faultyAlarms.any() == False and max_solarkwh_month > 0.4 * avg_solarTarget:
                    if solarkwh[l] < solarkwh[l - 1] * 0.7:  # Routine for Weather is not provided
                        miscount = 1
                        daynum = i
                        break
            solar_till_dynum = np.mean([solarkwh[:daynum]])
            solar_till_end = np.mean([solarkwh[:daynum]])
            if faultycount == 1 and solar_till_dynum > solar_till_end * 0.7:
                filed_challenges.append('Solar Missing6')

    return planning, filed_challenges



# Main Routine ----------------------------------------------------------------
# =================================================================

data = get_from_database()
final_data = sorting_data(data)

labelData_dictList = []

for i, single_data in enumerate(final_data):
    filed_challenges_cm = []
    planning_cm = []
    external_factors = []
    status_Label = []

    siteid = single_data[0][0]
    date = data_list(1, single_data)
    solarkw = data_list(2, single_data)
    solarkwhtarget = data_list(3, single_data)
    actualload = data_list(4, single_data)
    solarLoss = [random.random() for _ in range(len(date))]
    weather = random.choices(['cloudy', 'sunny'], k=len(date))
    faultyAlarms = np.full(len(date), False, dtype=bool)
    missingAlarms = np.full(len(date), False, dtype=bool)
    #     faultyAlarms = random.choices([True, False], k=len(date))
    #     missingAlarms = random.choices([True, False], k=len(date))
    one_site_data = np.array(
        [date, solarkw, solarkwhtarget, actualload, solarLoss, faultyAlarms, missingAlarms, weather],
        dtype='object').tolist()

    date_cm, solarkwh_cm, solarkwh_flag_cm, solartargetkwh_cm, solartargetkwh_flag_cm, actualload_cm, actualload_flag_cm, solarloss_cm, solarloss_flag_cm, faultyAlarm_cm, faultyAlarm_flag_cm, missingAlarms_cm, missingAlarms_flag_cm, weather_cm, weather_flag_cm, max_solarkwh_cm, avg_solarkwh_cm, avg_solartargetkwh_cm, avg_actualload_cm, avg_solarloss_cm = lastmonthdata_cal(
        1, one_site_data)
    date_pm, solarkwh_pm, solarkwh_flag_pm, solartargetkwh_pm, solartargetkwh_flag_pm, actualload_pm, actualload_flag_pm, solarloss_pm, solarloss_flag_pm, faultyAlarm_pm, faultyAlarm_flag_pm, missingAlarms_pm, missingAlarms_flag_pm, weather_pm, weather_flag_pm, max_solarkwh_pm, avg_solarkwh_pm, avg_solartargetkwh_pm, avg_actualload_pm, avg_solarloss_pm = lastmonthdata_cal(
        2, one_site_data)
    max_solarkwh_fromStart = max(solarkwh_cm + solarkwh_pm, default=0)

    if avg_solarkwh_cm != 0:
        solarimpact = (avg_solartargetkwh_cm - avg_solarkwh_cm) / (avg_solarkwh_cm * 100)
    else:
        solarimpact = None

    flags_cm = [solarkwh_flag_cm, solartargetkwh_flag_cm, actualload_flag_cm, solarloss_flag_cm, faultyAlarm_flag_cm,
                missingAlarms_flag_cm, weather_flag_cm]
    flags_comment = ['SolarkwhCM',
                     'SolarkwhtargetCM',
                     'actualloadkwCM',
                     'solarlossCM',
                     'faultyAlarmsCM',
                     'missingAlarmsCM',
                     'weatherCM']
    missing = []
    for i, b in enumerate(flags_cm):
        if b:
            missing.append(flags_comment[i])


    if avg_solarkwh_cm == 0:
        status_Label.append('--')
    elif avg_solarkwh_cm > avg_solartargetkwh_cm * 0.90:
        status_Label.append('Good Site')
    else:
        status_Label.append('Faulty Site')
        planning_cm, filed_challenges_cm = solarPresent(date_cm, solarkwh_cm, avg_solarkwh_cm, avg_solartargetkwh_cm,
                                                        avg_actualload_cm, avg_solarloss_cm, max_solarkwh_fromStart,
                                                        max_solarkwh_cm, faultyAlarm_cm, missingAlarms_cm)
    rca = {
        'solarImpact': solarimpact,
        'Field Challenges': filed_challenges_cm,
        'Infrastructure': planning_cm,
        'External Factors': None,
        'Missing Data': missing
    }

    labeled_site = {'siteid': siteid,
                    'solarkwhCMdata': solarkwh_cm,
                    'solarkwhCM': avg_solarkwh_cm,
                    'solarkwhtargetCMdata': solartargetkwh_cm,
                    'solarkwhtargetCM': avg_solartargetkwh_cm,
                    'actualloadkwhCMdata': actualload_cm,
                    'actualloadkwhCM': avg_actualload_cm,
                    'solarlossCMdata': solarloss_cm,
                    'solarlossCM': avg_solarloss_cm,
                    'weatherCM': weather_cm,
                    'faultyAlarmsCM': faultyAlarm_cm,
                    'missingAlarmsCM': missingAlarms_cm,
                    'solarkwhPM': avg_solarkwh_pm,
                    'solarkwhtargetPM': avg_solartargetkwh_pm,
                    'actualloadkwhPM': avg_actualload_pm,
                    'solarlossPM': avg_solarloss_pm,
                    'STATUS': status_Label,
                    'RCA': rca
                    }

    labelData_dictList.append(labeled_site)

# Converting Data to CSV format
labeled_dataframe = pd.DataFrame(labelData_dictList)
labeled_dataframe.to_csv('ProgressReport.csv', index=False)