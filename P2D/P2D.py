import numpy as np
from P2D.utils import pad_data_with_nans
import pandas as pd

class P2D:

    def __init__(self, input_data=None, model_type='inelastic', 
                 include_back_propagation = False, 
                 degree_resolution=0.5, COR=0,
                 propagation_time = '7d'):
        
        self.input_data = input_data
        self.model_type=model_type
        self.degree_resolution=degree_resolution
        self.COR = COR
        self.propagation_time = propagation_time
        
        if input_data is None:
            self.model = np.nan
        else:
            self.model = self.propagation()


        return
    
    def propagation(self):

        self.scale_to_degree_resolution()
        #output = self.input_data['V'].values*2

        n = int(np.floor(pd.Timedelta(self.propagation_time) / pd.Timedelta(self.cadence)))
        self.n = n # PROPAGATION STEPS

        hours = pd.Timedelta(self.cadence).total_seconds() / 3600
        L = pd.Timedelta(self.cadence).total_seconds() * 600 / 1.5e8 /10 # CHARACTERISTIC DISTANCE /10

        self.scaled_input_data['Index'] = range(len(self.scaled_input_data))
        self.scaled_input_data['Sim_step'] = self.scaled_input_data['Index'].values*0

        model_input = self.scaled_input_data[['N', 'V', 'R', 'CARR_LON_RAD', 'Index', 'Sim_step']].to_numpy()
        m = len(model_input) # INPUT DATA LENGTH
        #print(n, m)
        #model_output = np.zeros((n * (n + 1) // 2, 6))
        model_output = np.zeros((n * m, 6)) # CAREFUL, DONT GET ZERO AS A RESULT


        for i in range( n + m ):

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
                    #print(f'PHASE 2: m<n, i ={i}')
                    #continue
                    # first = m * (m + 1) // 2 + (i-m)*m
                    # last = first + m
                    # first_previous = first - m
                    # last_previous = last - m
                    first += m
                    last += m
                    first_previous += m-1
                    last_previous += m-1
                    #print(f'PHASE 2 m<n: old {last_previous-first_previous} new {last - first}')
                    #continue
                
                elif (m > n) and m > i >= n:
                    no_update=False
                    #continue
                    first += n
                    last += n
                    first_previous += n
                    last_previous += n
                    #print(f'PHASE 2 m>n: old {last_previous-first_previous} new {last - first}')
                    #continue

                ## PHASE 3: LAST TRIANGLE; DESCENDING
                elif i >= max(n, m):
                    # PHASE 3: Descending triangle
                    no_update = True
                    reverse_index = n + m - i - 1

                    first += reverse_index
                    last = first + reverse_index - 1
                    first_previous = first - (reverse_index + 1)
                    last_previous = first - 1
                    #print(f'PHASE 3 :{i} old {last_previous-first_previous} new {last - first}, and {first}, {last}, {first_previous}, {last_previous}')
                    

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
                model_output[first : last + 1, 5] = range(last  - first, -1, -1) #NEEDS TO BE ADAPTED FOR PHASE 3

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



        return merged_df
    

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

        self.scaled_input_data = scaled_input_data

        return 


    def observe_from(self, observer = 'Earth'):

        timeseries = self.scaled_input_data
        return timeseries