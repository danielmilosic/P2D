import numpy as np
from P2D.utils import pad_data_with_nans, spacecraft_ID, suppress_output
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from P2D import dataloader
from tqdm.notebook import tqdm
from P2D import get_coordinates
from typing import Tuple

class P2D:

    def __init__(self, input_data=None, model_type:str = 'inelastic', 
                 include_back_propagation:bool = False, 
                 degree_resolution:float = 0.5, COR:float = 0,
                 propagation_time:str = '7d',
                 coordinate_system:str = 'HEEQ',
                 timerange:list[str]=None,
                 smooth:str=None) -> None:
        """
        Initializes the P2D class.

        Parameters:
            input_data: pd.DataFrame, str, or list
                - A single DataFrame of spacecraft data
                - A spacecraft name string to auto-load data (requires timerange)
                - A list of spacecraft names or DataFrames

            model_type: str
                - 'inelastic' or 'ballistic'. Defaults to 'inelastic'.
                - Any other value defaults to 'ballistic'.

            include_back_propagation: bool
                - Whether to include backpropagated solar wind

            degree_resolution: float
                - Resolution in degrees for longitudinal and radial bins

            COR: float
                - Coefficient of restitution (for inelastic model only)

            propagation_time: str
                - Duration to propagate parcels, e.g., '7d'

            coordinate_system: str
                - Either 'HEEQ' or 'CARRINGTON'

            timerange: list of str
                - Required if loading spacecraft data by name

            smooth: str
                - Optional smoothing time window

        Output:
            This function doesn't return anything, but runs the model and saves the result as self.model
        """

        # save input parameters
        self.model_type=model_type
        self.degree_resolution=degree_resolution
        self.COR = COR
        self.propagation_time = propagation_time
        self.coordinate_system = coordinate_system
        self.list_of_timeseries_cutouts = []
        self.smooth = smooth

        if isinstance(input_data, (pd.DataFrame)):  
            self.input_data=input_data
            self.scale_to_degree_resolution()
            self.model = self.propagation()

        elif isinstance(input_data, (str)):
            if timerange is None:
                raise TypeError('You need to enter a timerange if you want to load data automatically')
            else:
                input_data = dataloader.load(spacecraft=input_data, timerange=timerange)
                self.input_data=input_data
                self.Earth_lon, self.Earth_rad = self.get_lon(self.input_data)
                self.scale_to_degree_resolution()
                self.model = self.propagation()

        elif isinstance(input_data, (list)):     
            if all(isinstance(t, str) for t in input_data):
                frame = []
                spacecraft = input_data
                for i in range (len(input_data)):
                    data = dataloader.load(spacecraft=input_data[i], timerange=timerange)
                    frame.append(data)
                input_data = frame
            elif all(isinstance(t, pd.DataFrame) for t in input_data):
                input_data = input_data
            
            model_output_frame = []
            for i in tqdm(input_data, desc=f'propagating spacecraft data'):
                self.input_data=i
                self.scale_to_degree_resolution()
                if isinstance(i, (pd.DataFrame)):  
                    model = self.propagation()
                    model_output_frame.append(model)
            self.model = pd.concat(model_output_frame)

        else:
            raise TypeError('Cannot run model without input data')

        return
    
    def propagation(self) -> pd.DataFrame:

        """
        Runs the propagation algorithm. Is called at the initiation of the P2D class. 
        
        """

        n = int(np.floor(pd.Timedelta(self.propagation_time) / pd.Timedelta(self.cadence)))
        self.n = n # PROPAGATION STEPS

        hours = pd.Timedelta(self.cadence).total_seconds() / 3600
        L = pd.Timedelta(self.cadence).total_seconds() * 600 / 1.5e8 /10 # CHARACTERISTIC DISTANCE /10
        
        self.scaled_input_data['Time'] = self.scaled_input_data.index
        self.scaled_input_data['Index'] = range(len(self.scaled_input_data))
        self.scaled_input_data['Sim_step'] = self.scaled_input_data['Index'].values*0

        model_input = self.scaled_input_data[['N', 'V', 'R', 'CARR_LON_RAD', 'Index', 'Sim_step']].to_numpy()
        m = len(model_input) # INPUT DATA LENGTH

        model_output = np.zeros((n * m, 6)) # CAREFUL, DONT GET ZERO AS A RESULT

        for i in tqdm(range(max(n,m))):
        #for i in tqdm(range( n + m )):

            if i == 0:
                # Initial values for the first step
                model_output[0, 0] = model_input[0, 0]  # 'N' column
                model_output[0, 1] = model_input[0, 1]  # 'V' column
                model_output[0, 2] = model_input[0, 2]  # 'R' column
                model_output[0, 3] = model_input[0, 3]  # 'CARR_LON_RAD' column
                model_output[0, 4] = model_input[0, 4]  # Index column
                model_output[0, 5] = model_input[0, 5]  # Sim_step column

            else:
                # Update values based on previous step and input data

                ## PHASE 1: INCREASING TRIANGLE
                if i < min(n,m):
                    no_update=False

                    first = i * (i + 1) // 2
                    last = first + i
                    first_previous = i * (i - 1) // 2
                    last_previous = first_previous + i + 1

                ## PHASE 2: ALL TAG ALONG
                elif (m < n) and m <= i < n:
                    no_update=True

                    first += m
                    last += m
                    first_previous += m-1
                    last_previous += m-1

                
                elif (m > n) and m > i >= n:
                    no_update=False

                    first += n
                    last += n
                    first_previous += n
                    last_previous += n


                ## PHASE 3: LAST TRIANGLE; DESCENDING
                elif i >= max(n, m):
                    
                    continue
                    no_update = True
                    reverse_index = n + m - i - 1

                    first += reverse_index
                    last = first + reverse_index - 1
                    first_previous = first - (reverse_index + 1)
                    last_previous = first - 1
                   

                #N
                model_output[first : last + 1, 0] = model_output[first_previous : last_previous, 0]
                if not no_update: model_output[last, 0] = model_input[i, 0]
                

                # N decreases quadratically
                #ratio = np.nan_to_num((model_output[first_previous : last_previous, 2]/model_output[first : last + 1, 2]), nan=1.0)
                #model_output[first : last + 1, 0] *= ratio**2


                #V
                model_output[first : last + 1, 1] = model_output[first_previous : last_previous, 1]
                if not no_update: model_output[last, 1] = model_input[i, 1]


                #R
                if type == 'reverse':
                    model_output[first : last + 1, 2] = model_output[first_previous : last_previous, 2] - \
                                                model_output[first_previous : last_previous, 1] / 1.4959787 / 10**8 * hours*3600.
                
                else:
                    model_output[first : last + 1, 2] = model_output[first_previous : last_previous, 2] + \
                                                model_output[first_previous : last_previous, 1] / 1.4959787 / 10**8 * hours*3600.
                if not no_update: model_output[last, 2] = model_input[i, 2]


                #CARR_LON_RAD
                if type == 'reverse':
                    model_output[first : last + 1, 3] = model_output[first_previous : last_previous, 3] + 0.0096 * hours
                else:
                    model_output[first : last + 1, 3] = model_output[first_previous : last_previous, 3] - 0.0096 * hours
                if not no_update: model_output[last, 3] = model_input[i, 3]
                
                #Index

                model_output[first : last + 1, 4] = model_output[first_previous : last_previous, 4]
                if not no_update: model_output[last, 4] = model_input[i, 4]


                #Sim_step
                model_output[first : last + 1, 5] = model_output[first_previous : last_previous, 5] +1

                if self.model_type == 'inelastic':
                    

                    # DON'T ALWAYS GO TO LOOP ? MAYBE MATRIX?

                    # R_now = model_output[first: last + 1, 2]
                    # L_now = model_output[first: last + 1, 3]
                    # R_past = model_output[:first, 2]
                    # L_past = model_output[:first, 3]

                    # Iterate over previous steps for momentum conservation
                    for j in range(first, last + 1):

                        # current_mass = model_output[j, 0]
                        # current_velocity = model_output[j, 1]
                        # current_R = model_output[j, 2]
                        # current_L = model_output[j, 3]

                        # past_data = model_output[:j]
                        # delta_R = np.abs(current_R - past_data[:, 2])  # IN AU
                        # delta_L = np.abs(current_L - past_data[:, 3])  # IN RAD

                        # mask = (delta_R < L) & (delta_L < self.degree_resolution / 180 * np.pi)
                        # num_colliding = np.sum(mask)
                        # print(num_colliding)

                        # if num_colliding > 0:

                        # WHICH PREVIOUS SHOULD BE  REALLY CONSIDERED? ONLY LAST MONTH WOULD BE NICE
                        delta_R = np.abs(model_output[j, 2] - model_output[:j, 2]) # IN AU
                        delta_L = np.abs(model_output[j, 3] - model_output[:j, 3]) # IN RAD

                        mask = (delta_R < L) & (delta_L  < self.degree_resolution/180*np.pi)  # Adjust the conditions as needed
                        if np.any(mask):
                            
                            print('COLLIDING')
                            # p_b = u_b * m_b (SUM)
                            pastsum = np.sum(model_output[0 : j][mask, 1] * model_output[0 : j][mask, 0])

                            # v_a = p_b + p_a / m_a + m_b(SUM)

                            # v_a = (1+COR)p_b + p_a - u_a*m_b(SUM)*COR  / (m_a + m_b(SUM))
                            model_output[j, 1] = (((self.COR+1.)*pastsum 
                                                + model_output[j, 1] * model_output[j, 0] 
                                                - model_output[j, 1]*np.sum(model_output[0 : j][mask, 0])*(self.COR)) /
                                                (model_output[j, 0] + np.sum(model_output[0 : j][mask, 0])))


                            # p_b = u_b * m_b (VECTOR)
                            past = model_output[0 : j][mask, 1] * model_output[0 : j][mask, 0]


                            # v_b = p_b + (1+COR)p_a - u_b(VECTOR)*m_a*COR / m_a + m_b(VECTOR)
                            model_output[0 : j][mask, 1] = ((past 
                                                        + model_output[j, 1] * model_output[j, 0]*(self.COR+1.)
                                                        - model_output[0 : j][mask, 1] * model_output[j, 0] * self.COR) /
                                                                    (model_output[j, 0] + model_output[0 : j][mask, 0]))
                            

        model_output = model_output[~np.all(model_output == 0, axis=1)]

        # Convert to DataFrame with column names
        output_df = pd.DataFrame(model_output, columns=['N', 'V', 'R', 'CARR_LON_RAD', 'Index', 'Sim_step'])

        # Merge with original (scaled) input data to reinsert untouched columns
        untouched_columns = [col for col in self.scaled_input_data.columns if col not in output_df.columns]
        merged_df = output_df.merge(
            self.scaled_input_data[['Index'] + untouched_columns],
            on='Index',
            how='left'
        )

        merged_df['Start_time_of_parcel'] = merged_df['Time']
        merged_df['Time'] = merged_df['Time'] +  pd.to_timedelta(self.cadence)*merged_df['Sim_step']

        return merged_df.set_index('Time')
    
  
    def get_lon(self, observer:str = 'OMNI', time:str = None) -> Tuple[float, float]:

        if time is None:
            time = self.model.index[-1]

        data = dataloader.load(spacecraft=observer, timerange=time)
        nearest_idx = -1
        if len(data)>0:
            nearest_idx = data.index.get_indexer([time], method='nearest')[0]
        if (nearest_idx == -1):
            print(f'{time} NO DATA, DOWNLOADING {observer} POSITION')
            lon, rad = suppress_output(get_coordinates.download_carrington_coordinates, dates=time, observer=observer)
            return lon[0][0]*np.pi/180, rad[0]
        else:
            if (data.index[nearest_idx] - time) > pd.Timedelta('1h'):
                print(f'{time} NO DATA, DOWNLOADING {observer} POSITION')
                lon, rad = suppress_output(get_coordinates.download_carrington_coordinates, dates=time, observer=observer)
                return lon[0][0]*np.pi/180, rad[0]
        
        # Use that index to retrieve the values
        lon = data.iloc[nearest_idx]['CARR_LON_RAD']
        rad = data.iloc[nearest_idx]['R']

        return lon, rad



    def scale_to_degree_resolution(self) -> None:

        # Constants
        degperhour = 0.55 # Solar rotation
        minutes_per_day = 1440

        # Calculate raw cadence
        raw_cadence_hours = self.degree_resolution / degperhour
        raw_cadence_minutes = raw_cadence_hours * 60  # Convert to minutes

        # Calculate the cadence as a divisor of minutes_per_day
        divisors = [d for d in range(1, minutes_per_day + 1) if minutes_per_day % d == 0]
        cadence_minutes = min(divisors, key=lambda x: abs(x - raw_cadence_minutes))
        cadence = f'{cadence_minutes}min'
        self.cadence = cadence

        data_filtered = self.input_data.loc[~self.input_data.index.isna()].dropna(subset=['V', 'N'])
        scaled_input_data = pad_data_with_nans(data_filtered.resample(rule=cadence).median()
                                        ,self.input_data.index[0]
                                        ,self.input_data.index[-1], cadence=cadence)

        scaled_input_data = scaled_input_data[~scaled_input_data.index.duplicated(keep='first')]
        if self.smooth is not None:
            columns = list(scaled_input_data.columns)
            columns.remove('CARR_LON_RAD')
            columns.remove('CARR_LON')
            scaled_input_data[columns] = scaled_input_data[columns].rolling(self.smooth).mean()
        self.scaled_input_data = scaled_input_data

        return 

    def observe_from(self, observer:str='OMNI', time_window:str='max') -> None:
        # Filter and sort model
        model = self.model[self.model['Sim_step'] == 1].sort_index()
        model = model[~model.index.duplicated(keep='first')]
        start_time = model.index[0].strftime('%Y-%m-%d-%H:%M')
        end_time = model.index[-1].strftime('%Y-%m-%d-%H:%M')

        # Load spacecraft data
        spacecraft_data = dataloader.load(spacecraft=observer, timerange=[start_time, end_time])
        spacecraft_data = spacecraft_data[~spacecraft_data.index.duplicated(keep='first')]

        unique_times = model.index  # already unique and sorted
        time_series = pd.DataFrame(index=unique_times)

        # Interpolate spacecraft data
        spacecraft_interp = (
            spacecraft_data
            .reindex(spacecraft_data.index.union(unique_times))
            .sort_index()
            .interpolate(method='time')
            .reindex(unique_times)
        )

        # Identify all spacecraft used in the model
        origin_spacecraft = model['Spacecraft_ID'].unique()

        # Pre-filter model by spacecraft ID to avoid repeated filtering
        model_by_sc = {
            sc: self.model[self.model['Spacecraft_ID'] == sc].reset_index()
            for sc in origin_spacecraft
        }

        best_model_rows = []

        for time in tqdm(unique_times):
            sc_row = spacecraft_interp.loc[time]

            for sc, model_rows in model_by_sc.items():
                # Subset of model_rows up to current time
                if time_window == 'max':
                    candidate_rows = model_rows[model_rows['Time'] <= time].copy()
                else:
                    candidate_rows = model_rows[(model_rows['Time'] <= time) & (model_rows['Time'] >= time-pd.to_timedelta(time_window))].copy()

                if candidate_rows.empty:
                    continue

                # Compute distances
                lon_distance = sc_row['R'] * (sc_row['CARR_LON_RAD'] - candidate_rows['CARR_LON_RAD'])
                r_distance = sc_row['R'] - candidate_rows['R']
                total_distance = np.sqrt(lon_distance**2 + r_distance**2)

                candidate_rows['Total_distance'] = total_distance

                best_idx = total_distance < 0.01 * self.degree_resolution/0.5

                if best_idx.any():
                    best_row = candidate_rows.loc[best_idx].copy()
                    best_row['Encounter_time'] = time
                    best_row['CARR_LON_RAD'] = sc_row['CARR_LON_RAD']
                    best_row['CARR_LON'] = sc_row['CARR_LON']

                    best_model_rows.append(best_row)

        time_series = pd.DataFrame(pd.concat(best_model_rows))
        time_series['Observer_SC'] = spacecraft_ID(observer, ID_number=True)
        
        # REMOVE DUPLICATES
        #print('REMOVING DUPLICATES')
        if len(time_series) > 0:
            start = time_series['Encounter_time'].min()
            end = time_series['Encounter_time'].max()

            if end - start > pd.Timedelta('25d'):
                rotations = ((end - start) // pd.Timedelta('25d')) + 1  # Integer Carrington segments
                deduped_segments = []

                for i in range(rotations):
                    rotation_start = start + i * pd.Timedelta('25d')
                    rotation_end = start + (i + 1) * pd.Timedelta('25d')

                    mask = (time_series['Encounter_time'] >= rotation_start) & (time_series['Encounter_time'] < rotation_end)
                    segment = time_series.loc[mask].copy().sort_values(by='Total_distance', ascending=True)

                    deduped = segment[~segment.duplicated(
                        subset=[
                                'V'
                                , 'Index'
                                , 'Spacecraft_ID'
                                , 'Start_time_of_parcel'
                                ], keep='first'
                    )].copy()



                    deduped_segments.append(deduped)

                # Rebuild deduplicated time_series
                time_series = pd.concat(deduped_segments).sort_values('Encounter_time').reset_index(drop=True)

            else:
                time_series = time_series.copy().sort_values(by='Total_distance', ascending=True)
                deduped = time_series[~time_series.duplicated(
                        subset=[
                                'V'
                                , 'Index'
                                , 'Spacecraft_ID'
                                , 'Start_time_of_parcel'
                                ], keep='first'
                    )].copy()
                time_series = deduped.sort_values('Encounter_time').reset_index(drop=True)

            self.list_of_timeseries_cutouts.append(time_series.set_index('Time'))
        return


    def plot(self, model=None, rlim=1.2, s=10, variable_to_plot='V', 
             xlim=None, fighandle=np.nan, axhandle=np.nan, dark_mode=False) -> None:
        
        # if no fig and axis handles are given, create a new figure
        if isinstance(fighandle, float):
            fig, axes = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        else:
            fig = fighandle
            axes = axhandle

        if dark_mode:
            plt.style.use('dark_background')
            sns.set_style("darkgrid", {'axes.facecolor': 'black'})
            fig.patch.set_facecolor('black')
            axes.set_facecolor('black')
        else:
            plt.style.use('default')
            sns.set_style("whitegrid")

        label_color = 'white' if dark_mode else 'black'


        if model is None:
            model = self.model.copy().sort_values('Start_time_of_parcel')
        else:
            model = model.copy().sort_values('Start_time_of_parcel')  

        Earth_lon, Earth_rad = self.get_lon(observer = 'OMNI', time=model.index.max())

        if self.coordinate_system == 'HEEQ':
            model['CARR_LON_RAD'] = model['CARR_LON_RAD'] - Earth_lon



        model = model.dropna(subset=[variable_to_plot])
        vmin=np.min(model[variable_to_plot])
        vmax=np.max(model[variable_to_plot])

        if variable_to_plot == 'V':
            sns.scatterplot(data=model, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue=variable_to_plot, palette='flare', hue_norm=(400,600), linewidth=0, legend=False)
        else: 
            sns.scatterplot(data=model, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue=variable_to_plot, palette='flare'
                            , hue_norm=(vmin, vmax), linewidth=0, legend=False)
        
        if self.coordinate_system == 'HEEQ':
            if (model['Spacecraft_ID']==6).any() > 0:
                sns.scatterplot(x= [0], y = Earth_rad, ax = axes, s=100, color='blue', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==1).any() > 0:
                psp_lon, psp_rad = self.get_lon(observer='PSP', time=model.index.max())
                sns.scatterplot(x= [(psp_lon - Earth_lon)], y = [psp_rad], ax = axes, s=100, color='red', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==2).any() > 0:
                solo_lon, solo_rad = self.get_lon(observer='SolO', time=model.index.max())
                sns.scatterplot(x= [(solo_lon - Earth_lon)], y = [solo_rad], ax = axes, s=100, color='yellow', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==4).any() > 0:
                stereo_a_lon, stereo_a_rad = self.get_lon(observer='STEREO-A', time=model.index.max())
                sns.scatterplot(x= [(stereo_a_lon - Earth_lon)], y = [stereo_a_rad], ax = axes, s=100, color='black', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==7).any() > 0:
                maven_lon, maven_rad = self.get_lon( observer='MAVEN', time=model.index.max())
                sns.scatterplot(x= [(maven_lon - Earth_lon)], y = [maven_rad], ax = axes, s=100, color='darkred', linewidth=0, legend=False)
            
        else:
            if (model['Spacecraft_ID']==6).any() > 0:
                sns.scatterplot(x= [Earth_lon], y = [Earth_rad], ax = axes, s=100, color='blue', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==1).any() > 0:
                psp_lon, psp_rad = self.get_lon(observer='PSP', time=model.index.max())
                sns.scatterplot(x= [psp_lon], y = [psp_rad], ax = axes, s=100, color='red', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==2).any() > 0:
                solo_lon, solo_rad = self.get_lon(observer='SolO', time=model.index.max())
                sns.scatterplot(x= [solo_lon], y = [solo_rad], ax = axes, s=100, color='yellow', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==4).any() > 0:    
                stereo_a_lon, stereo_a_rad = self.get_lon(observer='STEREO-A', time=model.index.max())
                sns.scatterplot(x= [stereo_a_lon], y = [stereo_a_rad], ax = axes, s=100, color='black', linewidth=0, legend=False)
            if (model['Spacecraft_ID']==7).any() > 0:
                maven_lon, maven_rad = self.get_lon(observer='MAVEN', time=model.index.max())
                sns.scatterplot(x= [maven_lon], y = [maven_rad], ax = axes, s=100, color='darkred', linewidth=0, legend=False)
        

        if xlim is not None:
            xlim = [xlim[0] * np.pi / 180, xlim[1] * np.pi / 180]
            axes.set_xlim(xlim)
        axes.set_rlim([0, rlim])
        axes.set_xlabel('')
        axes.set_ylabel('                                             longitude [°]', color=label_color)
        #axes.text(0.6, 0.5, '    r [AU]')
        #axes.legend(loc='lower center', bbox_to_anchor=(0.65, 0), ncol=1)
        axes.set_axisbelow(False)
        axes.grid(True, which='both', zorder=2, linewidth=0.2)
        axes.tick_params(colors=label_color)

        # Add a colorbar
        if variable_to_plot == 'V':
            sm = plt.cm.ScalarMappable(cmap='flare', norm=plt.Normalize(vmin=400, vmax=600))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, shrink=0.4, aspect=15)
            cbar.set_label('v [km/s]')
        else:
            sm = plt.cm.ScalarMappable(cmap='flare', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, shrink=0.4, aspect=15)
            cbar.set_label(variable_to_plot)

        # After colorbar creation:
        cbar.ax.xaxis.label.set_color(label_color)
        cbar.ax.tick_params(colors=label_color)


        if isinstance(fighandle, float):
                plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
                plt.show()
                plt.close()


    def plot_timeseries(self, model:pd.DataFrame=None, 
                        s:int=10, variable_to_plot:str='V', 
                        fighandle=np.nan, axhandle=np.nan, dark_mode:bool=False, 
                        vline:str=None, plot_against_time:bool=False) -> None:
        
        # if no fig and axis handles are given, create a new figure
        if isinstance(fighandle, float):
            fig, axes = plt.subplots(figsize=(10, 5))
        else:
            fig = fighandle
            axes = axhandle

        if dark_mode:
            plt.style.use('dark_background')
            sns.set_style("darkgrid", {'axes.facecolor': 'black'})
            fig.patch.set_facecolor('black')
            axes.set_facecolor('black')
        else:
            plt.style.use('default')
            sns.set_style("whitegrid")

        label_color = 'white' if dark_mode else 'black'

        if model is None:
            model = self.model.copy().sort_values('Start_time_of_parcel')
        else:
            model = model.copy().sort_values('Start_time_of_parcel')  

        Earth_lon, Earth_rad = self.get_lon(observer = 'OMNI', time=model.index.max())

        xmin = 360
        xmax = 0
        if self.coordinate_system == 'HEEQ':
            model['CARR_LON'] = model['CARR_LON'] - Earth_lon/np.pi*180
            model['CARR_LON'] = ((model['CARR_LON'] + 180) % 360) - 180
            xmin = 180
            xmax = -180
            axes.set_xlabel('HEEQ_LON')

        x = 'CARR_LON'
        if plot_against_time:
            x = 'plot_time'
            if 'Observer_SC' in model.columns:
                model['plot_time'] = model['Encounter_time']
            else:
                model['plot_time'] = pd.to_datetime(model.index)


        custom_palette = {
                        6: 'blue',
                        7: 'darkred',
                        2: 'orange',
                        4: 'black',
                        1: 'red',
                    }     
        
        sns.scatterplot(data=model, x=x, y = variable_to_plot, ax = axes, s=5, hue = model['Spacecraft_ID'], palette=custom_palette, linewidth=0, legend=False)

        if variable_to_plot =='V':
            ymin = 300
            ymax = 800
        else:
            ymin=np.min(model[variable_to_plot])
            ymax=np.max(model[variable_to_plot])
        if not plot_against_time:
            axes.set_ylim(ymin,ymax)
            axes.set_xlim(xmin,xmax)

        if vline is not None:
            if plot_against_time:
                axes.vlines(x = model.index.max, ymin=ymin, ymax=ymax, color=custom_palette[spacecraft_ID(vline, ID_number=True)], label=vline)
            
            else:
                lon, rad = self.get_lon(observer=vline, time=model.index.max())
                if self.coordinate_system == 'HEEQ':
                    axes.vlines(x = (((lon - Earth_lon)*180/np.pi + 180) % 360) - 180, ymin=ymin, ymax=ymax, color=custom_palette[spacecraft_ID(vline, ID_number=True)], label=vline)
                else:
                    axes.vlines(x = (lon)*180/np.pi, ymin=ymin, ymax=ymax, color=custom_palette[spacecraft_ID(vline, ID_number=True)], label=vline)
            

    def movie(self, time_window:str='max', cadence:str='6h', frametrate:int=30, observers:list[str]=[], **kwargs) -> None:
        
        if len(self.list_of_timeseries_cutouts) >0:
            self.list_of_timeseries_cutouts = []
        if len(observers)>0:
            print(f'FIRST CUTTING OUT TIMESERIES FROM {observers} PERSPECTIVE')
            list_of_observer_dfs = []
            if isinstance(observers, str):
                observers=[observers]
            for observer in observers:
                print(observer)
                self.observe_from(observer=observer, time_window=time_window)
            list_of_observer_dfs = self.list_of_timeseries_cutouts


        model_df = self.model.reset_index(drop=False).copy()

        time_df = model_df.sort_values('Time')
        start_time = time_df['Time'].iloc[0]
        end_time = time_df['Time'].iloc[-1]
        cadence = pd.Timedelta(cadence)

        times = pd.date_range(start=start_time, end=end_time, freq=cadence)

        def update(i):
            
            plt.clf()  # Clear the previous frame

            # === POLAR PLOT ===
            polar_left_in = (fig_height - polar_size_in - lef_margin_in) / 2  
            polar_bottom_in = (fig_height - polar_size_in) / 2 # Center vertically

            # Convert to normalized coordinates
            polar_left = polar_left_in / fig_width
            polar_bottom = polar_bottom_in / fig_height
            polar_width = polar_size_in / fig_width
            polar_height = polar_size_in / fig_height

            # Add polar plot
            ax_polar = fig.add_axes([polar_left, polar_bottom, polar_width, polar_height], projection='polar')


            current_time = times[i]
            #print(current_time)
            if time_window == 'max':
                plotting_df = model_df[model_df['Time'] <= current_time]
            else:
                window = pd.Timedelta(time_window)
                window_start = current_time - window
                mask = (model_df['Time'] > window_start) & (model_df['Time'] <= current_time)
                plotting_df = model_df[mask]

            self.plot(model = plotting_df.set_index('Time'), axhandle=ax_polar, fighandle=fig, **kwargs)

            # === TIME SERIES PLOTS (Right of Polar, Vertically Centered) ===

            # Total height in inches of the stacked time series plots
            total_ts_height_in = n_axes * timeseries_height_in + (n_axes - 1) * spacing_in

            # Compute the bottom of the stacked plots to vertically center them with polar plot
            ts_stack_center_in = polar_bottom_in + polar_size_in / 2
            ts_stack_bottom_in = ts_stack_center_in - total_ts_height_in / 2

            # Time series axes width
            ts_width_in = 8  # you can tweak this value
            ts_left_in = polar_left_in + polar_size_in + 1  # add horizontal margin

            for i, observer_df in enumerate(list_of_observer_dfs):
                # Compute bottom for each time series axes
                ts_bottom_in = ts_stack_bottom_in + i * (timeseries_height_in + spacing_in)

                # Normalize coordinates
                ts_left = ts_left_in / fig_width
                ts_bottom = ts_bottom_in / fig_height
                ts_width = ts_width_in / fig_width
                ts_height = timeseries_height_in / fig_height

                ax_ts = fig.add_axes([ts_left, ts_bottom, ts_width, ts_height])

                # Plotting
                observer_df = observer_df.reset_index(drop=False).copy()
                timeseries_mask = (
                    (observer_df['Encounter_time'] > window_start)
                    & (observer_df['Encounter_time'] <= current_time)
                )
                if len(observer_df[timeseries_mask]) > 0:
                    
                    plot_ts_df = observer_df[timeseries_mask].set_index('Encounter_time').copy()
                    if current_time not in plot_ts_df.index:
                        plot_ts_df.loc[current_time] = np.nan

                    self.plot_timeseries(
                        model=plot_ts_df,
                        axhandle=ax_ts,
                        fighandle=fig,
                        vline=observers[i]
                    )

                    ax_ts.text(
                        0.01, 0.95, observers[i],
                        transform=ax_ts.transAxes,
                        fontsize=12,
                        verticalalignment='top',
                        horizontalalignment='left'
                    )

                fig.text(0.2, 0.05, current_time.strftime('%Y-%m-%d %H:%M:%S'), fontsize=13)
                fig.text(0.02, 0.95, 'Model: ' + str(self.model_type), fontsize=15)
                fig.text(0.02, 0.93, 'Propagation: ' + str(self.propagation_time), fontsize=15)
            
        # === FIGURE SETUP ===
        fig_width = 20  # in inches
        n_axes = len(observers)
        timeseries_height_in = 2.0
        polar_size_in = 8.0
        lef_margin_in = 1.0
        bottom_margin_in = 1.0
        spacing_in = 0.5  # vertical space between axes (inches)

        # Total figure height based on all elements
        fig_height = 10
        fig = plt.figure(figsize=(fig_width, fig_height))


        #anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / frametrate)
        progress_bar = tqdm(total=len(times), desc="Animating")

        def update_with_progress(i):
            update(i)
            progress_bar.update(1)

        anim = FuncAnimation(fig, update_with_progress, frames=len(times), interval=1000 / frametrate)

        # Save the animation as a movie file
        filepath = f"P2D_prop_{self.propagation_time}_window_{time_window}_{len(self.input_data)}_sc.mp4"
        anim.save(filepath, writer='ffmpeg')
        print('mp4 file written to ' + filepath)
        
        return