from math import nan
from matplotlib import colors
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
 
class data_quality_analyzer(object):
    
    def __init__(self,data):
        if not isinstance(data, str):
            raise(Exception('The input should be a  Excel or Csv file or a path to Excel or Csv file'))
        else:
            if data[-4:] == '.csv':
                df = pd.read_csv(data, engine='python')
                self.data = df
                
            elif data[-5:] == '.xlsx':
                df = pd.read_excel(data)
                self.data = df
                
            else: raise(Exception('The input should be a csv or excel file'))
            
    def incompleteness_checking(self):
        if list(self.data.isna().sum(axis=1)) == list(np.zeros(len(list(self.data.columns)))):
                    return 0,[]
        else :  
            _, integrity_list = self.integrity_checking()
            if integrity_list == []:
                isna_check = pd.DataFrame(self.data.isna().sum(axis=1),columns=['values'])
                incomplet = [i+2 for i in list(isna_check.query('values!=0').index)]
                race_incompleteness = (len(incomplet)/self.data.shape[0])*100
                missing_val = {k: (v/self.data.shape[0])*100 for k, v in sorted(self.data.isna().sum().to_dict().items(), key=lambda item: item[1],reverse=True)}
                return race_incompleteness, incomplet,missing_val
            else:
                new_data = self.data.drop_duplicates().reset_index()
                new_data = new_data.drop(columns='index')
                isna_check_data = pd.DataFrame(new_data.isna().sum(axis=1),columns=['values'])
                new_incomplet = [i+2 for  i in list(isna_check_data.query('values!=0').index)]
               
                new_race_incompleteness = (len(new_incomplet)/new_data.shape[0])*100
                new_data.to_excel('new_data.xlsx',index=False)
                
                missing_val = {k: (v/new_data.shape[0])*100 for k, v in sorted(new_data.isna().sum().to_dict().items(), key=lambda item: item[1],reverse=True)}
                
                
                return new_race_incompleteness, new_incomplet, missing_val
    
    def integrity_checking(self):
        if self.data.duplicated().sum()==0:
             return 0,[]
        else:
            integrity = [i+2  for  i in list(self.data[self.data.duplicated(keep='first')].index) ]
            race_integrity = (len(integrity)/self.data.shape[0])*100
            return  race_integrity,  integrity
     
         
    def duplication_incompleteness_checker(self,number_of_col = 5):
        """
         Return the list of integrity line if exist, race of duplication, list of incomplete line if exist, race of completenes,
         accuracy of the data.
        """
        
        race_integrity,  integrity_list = self.integrity_checking()
        incomplet_race, list_incompleteness,missing_value  = self.incompleteness_checking()   
        
        missing_val_part = {k: missing_value[k] for k in list(missing_value)[:number_of_col]}
                 
        
        if integrity_list == []:
            print('No duplication in the data') 
            print(f'The race of duplication is {race_integrity :.2f}%')
            print('--------------------------------------------------------', end='\n')
        else: 
            print(f'integrityd lines are: {integrity_list}') 
            print(f'The race of duplication is {race_integrity :2f}.%')
            print('--------------------------------------------------------', end='\n')
            
        if list_incompleteness== []:
            print('No imcompleteness in the data')
            print(f'The race of completeness is {incomplet_race :.2f}%')
            print('--------------------------------------------------------', end='\n')
        else:
            print(f'There is incompleteness at lines:{list_incompleteness}')
            print(f'The race of completeness is {incomplet_race :.2f}%')
            print('--------------------------------------------------------', end='\n')
  
         
        accuracy = 100- (race_integrity + incomplet_race)
        print(f'Accuracy of the data: {accuracy :.2f}% ')
        print('--------------------------------------------------------', end='\n')
        
        
        metrics = ['Incompleteness','Duplication','Accuracy']
        error_values = [incomplet_race,race_integrity, accuracy] 
        #pie_values = [len(list_incompleteness), len(integrity_list), self.data.shape[0] -(len(list_incompleteness)+len(integrity_list)),   (len(list_incompleteness)+len(integrity_list))/2 ]
        
        missing_value_col = list(missing_val_part.keys())
        missing_value_value = list(missing_val_part.values())
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        fig.add_trace(go.Bar(x=metrics, y=error_values,marker=dict( color = ['yellow','blue','#a0911b']), text= [ "{:.2f}%".format(val) for val in error_values],
                             textposition='outside'), row=1, col=1)

        #fig.add_trace(go.Pie(values=pie_values,labels=metrics,marker=dict( colors = ['yellow','blue','green','#a0911b']),text=metrics),
                   # row=1, col=2)
        fig.add_trace(go.Bar(x=missing_value_col, y=missing_value_value, text= [ "{:.2f}%".format(val) for val in missing_value_value],
                          textposition='outside'), row=1, col=2)
        fig.update_layout(height=500, showlegend=False)

        fig.show()
                
        
    def fill_missing_values(self):
        if self.data.shape[0] > 65000:
            raise(Exception('The size of the data should be less than 65000.'))
        
        elif list(self.data.isna().sum(axis=1)) == list(np.zeros(len(list(self.data.columns)))):
            raise(Exception('There is no missing value in the dataset'))
        
        else: 
            df_nan = self.data[self.data.isna().any(axis=1)].reset_index()
            df_nan = df_nan.drop(columns=['index'])
            
            df = self.data.dropna(axis=0).reset_index()
            df = df[df.columns.difference(['index'])]
            
            
            ann1 =  RandomForestRegressor(n_estimators=200,max_depth=10)
            #MLPRegressor(hidden_layer_sizes=15, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=6000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
            
        def train(df, df_nan):
            for col in list(df_nan.columns):
                if  df_nan.isna().sum()[col]!=0:
                    
                    y = df[col]
                    X = df.drop(columns=[col])
                    ann1.fit(X, y)
                    
                    x_test = df_nan.drop(columns=[col])
                    x_test_copy = x_test.copy()
                    x_test_copy[col] = ann1.predict(x_test)
                    x_test = x_test_copy
                    return x_test_copy   
                else: pass   
                  
        if df.shape[1] == 0:
               print('all the columns have missing values. This program is not appropriate')
        else:
            # fit the models to our data
            x_test_result = train(df,df_nan)
            init_df = pd.concat([x_test_result,df], axis=0,ignore_index=True)
            init_df.to_csv('result_dataset.csv',index=False)
            init_df.to_excel('result_dataset.xlsx',index=False)
        return init_df   
            
