import numpy as np
from P2D.utils import pad_data_with_nans
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from P2D import dataloader
from tqdm.notebook import tqdm

class P2D:

    def __init__(self, input_data=None, model_type='inelastic', 
                 include_back_propagation = False, 
                 degree_resolution=0.5, COR=0,
                 propagation_time = '7d',
                 coordinate_system = 'HEEQ',
                 timerange=None):
        
        #self.input_data = input_data
        self.model_type=model_type
        self.degree_resolution=degree_resolution
        self.COR = COR
        self.propagation_time = propagation_time
        self.coordinate_system = coordinate_system

        if input_data is None:
            self.model = np.nan
        elif isinstance(input_data, (str)):
            if timerange is None:
                raise TypeError('You need to enter a timerange if you want to load data automatically')
            else:
                input_data = dataloader.load(spacecraft=input_data, timerange=timerange)
        elif isinstance(input_data, (list)):     
            if all(isinstance(t, str) for t in input_data):
                frame = []
                for i in range (len(input_data)):
                    data = dataloader.load(spacecraft=input_data, timerange=timerange)
                    frame.append(data)
                input_data = pd.concat(frame)
            elif all(isinstance(t, pd.DataFrame) for t in input_data):
                input_data = pd.concat(input_data)
        
        self.input_data=input_data
        self.scale_to_degree_resolution()

        if isinstance(input_data, (pd.DataFrame)):  
            self.model = self.propagation()
        else:
            raise TypeError('Cannot run model without input data')

        return
    
    def propagation(self):

        #self.scale_to_degree_resolution()
        #output = self.input_data['V'].values*2

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


        for i in tqdm(range( n + m )):

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
                ratio = np.nan_to_num((model_output[first_previous : last_previous, 0]/model_output[first : last + 1, 0]), nan=1.0)
                model_output[first : last + 1, 0] *= ratio**2


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
                #model_output[first : last + 1, 5] = range(last  - first, -1, -1) #NEEDS TO BE ADAPTED FOR PHASE 2 AND 3
                model_output[first : last + 1, 5] = model_output[first_previous : last_previous, 5] +1

                if self.model_type == 'inelastic':

                    # Iterate over previous steps for momentum conservation
                    for j in range(first, last + 1):
                        delta_R = np.abs(model_output[j, 2] - model_output[:j, 2]) # IN AU
                        delta_L = np.abs(model_output[j, 3] - model_output[:j, 3]) # IN RAD

                        mask = (delta_R < L) & (delta_L  < self.degree_resolution/180*np.pi/1.5 )  # Adjust the conditions as needed
                        
                        if np.any(mask):
                            
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
        #merged_df.set_index('Time')

        return merged_df.set_index('Time')
    
    def get_lon(self, model=None, observer='OMNI'):
        
        if model is None:
            model = self.model[self.model['Sim_step']==1]
        else:
            model = model[model['Sim_step']<=1]

        model_df = model.sort_index()
        model_df = model_df[~model_df.index.duplicated(keep='first')]
        start_time = model_df.index[0].strftime('%Y-%m-%d-%H:%M')
        end_time = model_df.index[-1].strftime('%Y-%m-%d-%H:%M')

        spacecraft_data = dataloader.load(spacecraft=observer, timerange=[start_time,end_time])
        
        # Get unique model times sorted
        unique_times = model_df.index.sort_values()
        # Interpolate omni_data['CARR_LON_RAD'] at unique_times (time-based interpolation)
        spacecraft_interp = (spacecraft_data['CARR_LON_RAD']
                    .reindex(spacecraft_data.index.union(unique_times))
                    .sort_index()
                    .interpolate(method='time')
                    .reindex(unique_times))
        
        lon = spacecraft_interp[unique_times[-1]]
        return lon



    def scale_to_degree_resolution(self):

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
        self.scaled_input_data = scaled_input_data

        return 


    def observe_from(self, observer = 'Earth'):

        timeseries = self.scaled_input_data
        return timeseries
    

    def plot(self, model=None, rlim=1.2, s=10, variable_to_plot='V', xlim=None, fighandle=np.nan, axhandle=np.nan):
        
        if model is None:
            model = self.model.copy()

        if self.coordinate_system == 'HEEQ':
            Earth_lon = self.get_lon(model = model)
            model['CARR_LON_RAD'] = model['CARR_LON_RAD'] - Earth_lon

      
        # if no fig and axis handles are given, create a new figure
        if isinstance(fighandle, float):
            fig, axes = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        else:
            fig = fighandle
            axes = axhandle
            #fig.set_size_inches(10, 10)


        model = model.dropna(subset=[variable_to_plot])
        vmin=np.min(model[variable_to_plot])
        vmax=np.max(model[variable_to_plot])

        if variable_to_plot == 'V':
            sns.scatterplot(data=model, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue=variable_to_plot, palette='flare', hue_norm=(400,600), linewidth=0, legend=False)
        else: 
            sns.scatterplot(data=model, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue=variable_to_plot, palette='flare'
                            , hue_norm=(vmin, vmax), linewidth=0, legend=False)
        
        if xlim is not None:
            xlim = [xlim[0] * np.pi / 180, xlim[1] * np.pi / 180]
            axes.set_xlim(xlim)
        axes.set_rlim([0, rlim])
        axes.set_xlabel('')
        axes.set_ylabel('                                             longitude [°]')
        axes.text(0.6, 0.5, '    r [AU]')
        #axes.legend(loc='lower center', bbox_to_anchor=(0.65, 0), ncol=1)
        axes.set_axisbelow(False)
        axes.grid(True, which='both', zorder=3, linewidth=0.2)

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


        if isinstance(fighandle, float):
                plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
                plt.show()
                plt.close()


    def movie(self, time_window='max', cadence='6h', frametrate=30, **kwargs):

        model_df = self.model  
        model_df = model_df.reset_index(drop=False)

        time_df = model_df.sort_values('Time')
        start_time = time_df['Time'].iloc[0]
        end_time = time_df['Time'].iloc[-1]
        cadence = pd.Timedelta(cadence)

        times = pd.date_range(start=start_time, end=end_time, freq=cadence)
        frames = []


        def update(i):
            
            plt.clf()  # Clear the previous frame
            ax = fig.add_subplot(111, projection='polar')

            current_time = times[i]

            if time_window == 'max':
                plotting_df = model_df[model_df['Time'] <= current_time]
            else:
                window = pd.Timedelta(time_window)
                window_start = current_time - window
                mask = (model_df['Time'] > window_start) & (model_df['Time'] <= current_time)
                plotting_df = model_df[mask]

            self.plot(model = plotting_df.set_index('Time'), axhandle=ax, fighandle=fig, **kwargs)

        
        fig, axes = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
 
        #anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / frametrate)
        progress_bar = tqdm(total=len(times), desc="Animating")

        def update_with_progress(i):
            update(i)
            progress_bar.update(1)

        anim = FuncAnimation(fig, update_with_progress, frames=len(times), interval=1000 / frametrate)

        # Save the animation as a movie file
        filepath = f"P2D_prop_{self.propagation_time}_window_{time_window}.mp4"
        anim.save(filepath, writer='ffmpeg')
        print('mp4 file written to ' + filepath)
        
        return