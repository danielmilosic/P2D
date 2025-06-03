import pandas as pd
import numpy as np
import numba    
from CIRESA.utils import spacecraft_ID, pad_data_with_nans

def inelastic_radial(spacecraft, degree_resolution=0.5, COR=0):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    with momentum conservation, no energy conservation. Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation can (*has to) to contain the following columns:
            'N': density
            'V'*: velocity
            'R'*: distance
            'CARR_LON_RAD'*: carrington longitude in radians
            'Spacecraft_ID': 1-7
            'Region': identified CIR Region
            'TT': travel time

            cadence: the cadence with which the model runs

            COR: Coefficient of Resitution
                0: perfectly inelastic
                0<COR<1 : real inelastic
                1: perfectly elastic
    Returns:
    - sim: pd DataFrame

        columns: (N, V, R, CARR_LON_RAD, ITERATION, Region, Spacecraft_ID)
        ITERATION denotes the number of steps as well as the number of data points

    Examples:
        sim = inelastic_propagation(solo)
    """
    ID = spacecraft_ID(spacecraft, ID_number=True)

    # Constants
    degperhour = 0.55 # Solar rotation
    minutes_per_day = 1440

    # Calculate raw cadence
    raw_cadence_hours = degree_resolution / degperhour
    raw_cadence_minutes = raw_cadence_hours * 60  # Convert to minutes

    # Calculate the cadence as a divisor of minutes_per_day
    divisors = [d for d in range(1, minutes_per_day + 1) if minutes_per_day % d == 0]
    cadence_minutes = min(divisors, key=lambda x: abs(x - raw_cadence_minutes))
    cadence = f'{cadence_minutes}min'

    if 'Region' not in spacecraft:
        spacecraft['Region'] = spacecraft['V']*np.nan
    if 'Spacecraft' not in spacecraft:
        spacecraft['Spacecraft'] = spacecraft['V']*np.nan

    spacecraft = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region', 'Spacecraft_ID', 'LAT']]
    spacecraft_filtered = spacecraft.loc[~spacecraft.index.isna()].dropna(subset=['V', 'N'])
    spacecraft = pad_data_with_nans(spacecraft_filtered.resample(rule=cadence).median()
                                    ,spacecraft.index[0]
                                    ,spacecraft.index[-1], cadence=cadence)

    L = pd.Timedelta(cadence).total_seconds() * 600 / 1.5e8 /10  # CHARACTERISTIC DISTANCE /10

    hours = pd.Timedelta(cadence).total_seconds() / 3600

    input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region', 'Spacecraft_ID', 'LAT']].to_numpy()

    sim = radial_prop(input_data, type = 'inelastic', COR=COR
                                , L=L, hours=hours, degree_resolution=degree_resolution)

    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4],
        'Region': np.round(sim[:,5]),
        'Spacecraft_ID': ID,
        'TT': sim[:, 7] * cadence_minutes/60,
        'LAT': sim[:, 8]

    }, index=spacecraft.index[0] + sim[:, 7] * pd.Timedelta(cadence))

    return output_data


def ballistic(spacecraft, degree_resolution=0.5, COR=0):
    """
    Generate a new NumPy array with a simulated ballistic propagation of the spacecraft data. 
    Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation can (*has to) to contain the following columns:
            'N': density
            'V'*: velocity
            'R'*: distance
            'CARR_LON_RAD'*: carrington longitude in radians
            'Spacecraft_ID': 1-7
            'Region': identified CIR Region

            cadence: the cadence with which the model runs

            COR: Coefficient of Resitution
                0: perfectly inelastic
                0<COR<1 : real inelastic
                1: perfectly elastic
    Returns:
    - sim: pd DataFrame

        columns: (N, V, R, CARR_LON_RAD, ITERATION, Region, Spacecraft_ID)
        ITERATION denotes the number of steps as well as the number of data points

    Examples:
        sim = ballistic(solo)
    """

    ID = spacecraft_ID(spacecraft, ID_number=True)

    # Constants
    degperhour = 0.55 # Solar rotation
    minutes_per_day = 1440

    # Calculate raw cadence
    raw_cadence_hours = degree_resolution / degperhour
    raw_cadence_minutes = raw_cadence_hours * 60  # Convert to minutes

    # Calculate the cadence as a divisor of minutes_per_day
    divisors = [d for d in range(1, minutes_per_day + 1) if minutes_per_day % d == 0]
    cadence_minutes = min(divisors, key=lambda x: abs(x - raw_cadence_minutes))
    cadence = f'{cadence_minutes}min'

    if 'Region' not in spacecraft:
        spacecraft['Region'] = spacecraft['V']*np.nan
    if 'Spacecraft' not in spacecraft:
        spacecraft['Spacecraft'] = spacecraft['V']*np.nan

    spacecraft = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region', 'Spacecraft_ID', 'LAT']]
    
    spacecraft_filtered = spacecraft.dropna(subset=['V', 'N'])
    try:
        spacecraft = pad_data_with_nans(spacecraft_filtered.resample(rule=cadence).median()
                                        ,spacecraft.index[0]
                                        ,spacecraft.index[-1], cadence=cadence)
    except Exception:
        pass

    L = pd.Timedelta(cadence).total_seconds() * 600 / 1.5e8 /10 # CHARACTERISTIC DISTANCE /10

    hours = pd.Timedelta(cadence).total_seconds() / 3600

    input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region', 'Spacecraft_ID', 'LAT']].to_numpy()

    sim = radial_prop(input_data, type = 'ballistic', COR=COR
                                , L=L, hours=hours, degree_resolution=degree_resolution)

    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4],
        'Region': np.round(sim[:,5]),
        'Spacecraft_ID': np.round(sim[:,6]),
        'TT': sim[:, 7] * cadence_minutes/60,
        'LAT': sim[:, 8]

    }, index=spacecraft.index[0] + sim[:, 7] * pd.Timedelta(cadence))
    
    return output_data

def ballistic_reverse(spacecraft, degree_resolution=0.5, COR=0):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    with momentum conservation, no energy conservation. Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation can (*has to) to contain the following columns:
            'N': density
            'V'*: velocity
            'R'*: distance
            'CARR_LON_RAD'*: carrington longitude in radians
            'Spacecraft_ID': 1-7
            'Region': identified CIR Region

            cadence: the cadence with which the model runs

            COR: Coefficient of Resitution
                0: perfectly inelastic
                0<COR<1 : real inelastic
                1: perfectly elastic
    Returns:
    - sim: pd DataFrame

        columns: (N, V, R, CARR_LON_RAD, ITERATION, Region, Spacecraft_ID)
        ITERATION denotes the number of steps as well as the number of data points

    Examples:
        sim = ballistic_reverse(solo)
    """
    ID = spacecraft_ID(spacecraft, ID_number=True)

    # Constants
    degperhour = 0.55 # Solar rotation
    minutes_per_day = 1440

    # Calculate raw cadence
    raw_cadence_hours = degree_resolution / degperhour
    raw_cadence_minutes = raw_cadence_hours * 60  # Convert to minutes

    # Calculate the cadence as a divisor of minutes_per_day
    divisors = [d for d in range(1, minutes_per_day + 1) if minutes_per_day % d == 0]
    cadence_minutes = min(divisors, key=lambda x: abs(x - raw_cadence_minutes))
    cadence = f'{cadence_minutes}min'

    # Force cadence to be divisible by the number of minutes in a day
    try:
        cadence = str(minutes_per_day // round(minutes_per_day / raw_cadence_minutes))+'min'

    except:
        cadence = '2H'

    if 'Region' not in spacecraft:
        spacecraft['Region'] = spacecraft['V']*np.nan
    if 'Spacecraft' not in spacecraft:
        spacecraft['Spacecraft'] = spacecraft['V']*np.nan

    spacecraft = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region', 'Spacecraft_ID', 'LAT']]
    
    spacecraft_filtered = spacecraft.dropna(subset=['V', 'N'])
    spacecraft = pad_data_with_nans(spacecraft_filtered.resample(rule=cadence).median()
                                    ,spacecraft.index[0]
                                    ,spacecraft.index[-1], cadence=cadence)


    L = pd.Timedelta(cadence).total_seconds() * 600 / 1.5e8 /10 # CHARACTERISTIC DISTANCE /10

    hours = pd.Timedelta(cadence).total_seconds() / 3600

    input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region', 'Spacecraft_ID', 'LAT']].to_numpy()

    sim = radial_prop(input_data, type = 'reverse', COR=COR
                                , L=L, hours=hours, degree_resolution=degree_resolution)

    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4],
        'Region': np.round(sim[:,5]),
        'Spacecraft_ID': np.round(sim[:,6]),
        'TT': sim[:, 7] * cadence_minutes/60,
        'LAT': sim[:, 8]

    }, index=spacecraft.index[0] - sim[:, 7] * pd.Timedelta(cadence))
    
    return output_data


#@numba.jit(nopython=True, parallel = True)
def radial_prop( input_data, L, hours, COR, degree_resolution, type = 'inelastic'):
    """
    backbone of inelastic_radial_new
    """
    n = input_data.shape[0] -1

    # Pre-allocate array with NaN values for the entire structure
    sim = np.empty((n * (n + 1) // 2, 9))  # 9 columns for 'N', 'V', 'R', 'L', 'ITERATION', 'Region', 'Spacescraft_ID', 'TT', 'LAT'
    
    # Generate the iteration sequence 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, ...
    iteration_values = np.concatenate([np.full(i, i - 1) for i in range(1, n + 1)])
    sim[:, 4] = iteration_values

    # # Generate the iteration sequence n, n-1, n-1, n-2, n-2, n-2, n-3, n-3, n-3, n-3, ...
    # tt_values = np.concatenate([np.full(i, n - i) for i in range(1, n + 1)]) 
    # sim[:, 7] = tt_values

    tt_input =  np.linspace(n-1, 0, n)

    # Iterate over n steps for simulation
    for i in range(n):

        if i == 0:
            # Initial values for the first step
            #sim[0, 4] = 0 # ITERATION column
            sim[0, 1] = input_data[0, 1]  # 'V' column
            sim[0, 0] = input_data[0, 0]  # 'N' column
            sim[0, 2] = input_data[0, 2]  # 'R' column
            sim[0, 3] = input_data[0, 3]# - 0.0096 * hours  # 'CARR_LON_RAD' column

            sim[0, 5] = input_data[0, 4] # Region column
            sim[0, 6] = input_data[0, 5] # Spacecraft_ID column
            sim[0, 7] = tt_input[0] # TT column
            sim[0, 8] = input_data[0, 6]

        else:
            # Update values based on previous step and input data
            first = i * (i + 1) // 2
            last = first + i
            first_previous = i * (i - 1) // 2
            
            #ITERATION
            sim[first : last + 1, 4] += 1


            #V
            sim[first : last + 1, 1] = sim[first_previous : first_previous + i + 1, 1]
            sim[last, 1] = input_data[i, 1]


            #N
            sim[first : last + 1, 0] = sim[first_previous : first_previous + i + 1, 0]
            sim[last, 0] = input_data[i, 0]

            #LAT
            sim[first : last + 1, 8] = sim[first_previous : first_previous + i + 1, 8]
            sim[last, 8] = input_data[i, 6]

            #TT
            sim[first : last + 1, 7] = sim[first_previous : first_previous + i + 1, 7]
            sim[last, 7] = tt_input[i]


            #R
            if type == 'reverse':
                sim[first : last + 1, 2] = sim[first_previous : first_previous + i + 1, 2] - \
                                               sim[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * hours*3600.
            
            else:
                sim[first : last + 1, 2] = sim[first_previous : first_previous + i + 1, 2] + \
                                               sim[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * hours*3600.
            sim[last, 2] = input_data[i, 2]

            # N decreases quadratically
            ratio = np.nan_to_num((sim[first_previous : first_previous + i + 1, 0]/sim[first : last + 1, 0]), nan=1.0)
            #print(ratio)
            sim[first : last + 1, 0] *= ratio**2

            #CARR_LON_RAD
            if type == 'reverse':
                sim[first : last + 1, 3] = sim[first_previous : first_previous + i + 1, 3] + 0.0096 * hours
            else:
                sim[first : last + 1, 3] = sim[first_previous : first_previous + i + 1, 3] - 0.0096 * hours
            sim[last, 3] = input_data[i, 3]
            
            #REGION

            sim[first : last + 1, 5] = sim[first_previous : first_previous + i + 1, 5]
            sim[last, 5] = input_data[i, 4]

            # Spacecraft_ID

            sim[first : last + 1, 6] = sim[first_previous : first_previous + i + 1, 6]
            sim[last, 6] = input_data[i, 5]

            if type == 'inelastic':

                # Iterate over previous steps for momentum conservation
                for j in range(first, last + 1):
                    delta_R = np.abs(sim[j, 2] - sim[:j, 2]) # IN AU
                    delta_L = np.abs(sim[j, 3] - sim[:j, 3]) # IN RAD

                    mask = (delta_R < L) & (delta_L  < degree_resolution/180*np.pi/1.5 )  # Adjust the conditions as needed
                    
                    if np.any(mask):
                        
                        # p_b = u_b * m_b (SUM)
                        pastsum = np.sum(sim[0 : j][mask, 1] * sim[0 : j][mask, 0])

                        # v_a = p_b + p_a / m_a + m_b(SUM)

                        # v_a = (1+COR)p_b + p_a - u_a*m_b(SUM)*COR  / (m_a + m_b(SUM))
                        sim[j, 1] = (((COR+1.)*pastsum 
                                            + sim[j, 1] * sim[j, 0] 
                                            - sim[j, 1]*np.sum(sim[0 : j][mask, 0])*(COR)) /
                                            (sim[j, 0] + np.sum(sim[0 : j][mask, 0])))


                        # p_b = u_b * m_b (VECTOR)
                        past = sim[0 : j][mask, 1] * sim[0 : j][mask, 0]


                        # v_b = p_b + (1+COR)p_a - u_b(VECTOR)*m_a*COR / m_a + m_b(VECTOR)
                        sim[0 : j][mask, 1] = ((past 
                                                    + sim[j, 1] * sim[j, 0]*(COR+1.)
                                                    - sim[0 : j][mask, 1] * sim[j, 0] * COR) /
                                                                (sim[j, 0] + sim[0 : j][mask, 0]))


    deg = (sim[:,3] * 180/ np.pi) % 360
    sim[:,3] = deg * np.pi/180


    return sim


def cut_from_sim(sim, spacecraft=None):
    """
    Extracts the values of 'V' from the sim DataFrame that correspond to the closest 'CARR_LON_RAD'
    and 'R' values in the spacecraft DataFrame.
    
    Parameters:
    - sim: pd.DataFrame
        Simulation data with columns 'CARR_LON_RAD', 'R', and 'V', 'Region' optional.
    - spacecraft: pd.DataFrame, optional
        Spacecraft data with columns 'CARR_LON_RAD' and 'R'. If None, default values are 1AU.
    
    Returns:
    - result: pd.DataFrame
        DataFrame with the closest values of 'V' and 'Region' from sim based on 'CARR_LON_RAD' and 'R' in spacecraft.
    """
    
    if spacecraft is None:
        spacecraft = pd.DataFrame({
            'CARR_LON_RAD': np.linspace(0, 2*np.pi, 720),
            'R': np.ones(720)  # 1AU
        })

    if isinstance(spacecraft, (int, float)):
        spacecraft = pd.DataFrame({
            'CARR_LON_RAD': np.linspace(0, 2*np.pi, 720),
            'R': np.ones(720) * spacecraft # 1AU
        })
    
    if isinstance(spacecraft, pd.DataFrame):
        spacecraft = pd.DataFrame({
            'CARR_LON_RAD': spacecraft['CARR_LON_RAD'],
            'R': spacecraft['R'],  # 1AU
        })
        spacecraft.reset_index(drop=True, inplace=True)
    
    sim = sim.reset_index(drop=True)

    # Create an empty array to store the closest 'V' values
    closest_V = np.empty(len(spacecraft))
    closest_N = np.empty(len(spacecraft))
    closest_TT = np.empty(len(spacecraft))
    Region = np.zeros(len(spacecraft))
    Space_ID = np.zeros(len(spacecraft))
    Iteration = np.zeros(len(spacecraft))
    Lat = np.zeros(len(spacecraft))

    # Iterate over each row in the spacecraft DataFrame
    for i, row in spacecraft.iterrows():
        # Calculate the Euclidean distance between spacecraft values and sim values for 'CARR_LON_RAD' and 'R'
        distances = np.sqrt((sim['CARR_LON_RAD'] - row['CARR_LON_RAD'])**2 +
                            (sim['R'] - row['R'])**2)
        
        #print(distances)
        # Find the index of the minimum distance
        if len(distances)>0:
            closest_idx = distances.idxmin()
        else:
            closest_idx = np.nan
        # Store the corresponding 'V' value, Region and Spacecraft ID
        if not np.isnan(closest_idx):
            closest_V[i] = sim.loc[closest_idx, 'V']
            closest_N[i] = sim.loc[closest_idx, 'N']
            closest_TT[i] = sim.loc[closest_idx, 'TT']

            if 'Region' in sim:
                Region[i] = sim.loc[closest_idx, 'Region']
            if 'Spacecraft_ID' in sim:
                Space_ID[i] = sim.loc[closest_idx, 'Spacecraft_ID']
            if 'LAT' in sim:
                Lat[i] = sim.loc[closest_idx, 'LAT']

            Iteration[i] = sim.loc[closest_idx, 'ITERATION']
            
            if abs(sim.loc[closest_idx, 'CARR_LON_RAD'] - row['CARR_LON_RAD']) > 0.3 / 180 * np.pi:
                
                closest_V[i] = np.nan
                if 'Region' in sim:
                    Region[i] = 0
            
            if abs(sim.loc[closest_idx, 'R'] - row['R']) > 0.1:
                closest_V[i] = np.nan
                if 'Region' in sim:
                    Region[i] = 0

    # Create a new DataFrame with the results
    result = pd.DataFrame({
        'CARR_LON_RAD': spacecraft['CARR_LON_RAD'],
        'R': spacecraft['R'],
        'V': closest_V,
        'N': closest_N, 
        'Region': Region,
        'Spacecraft_ID': Space_ID,
        'ITERATION': Iteration,
        'TT': closest_TT,
        'LAT': Lat
    })
    
    return result.dropna(subset=['CARR_LON_RAD', 'R', 'V'])
