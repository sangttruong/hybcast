import pandas as pd
powerbi = pd.read_csv('powerbi-lite.csv')
qlik = pd.read_csv('qlik.csv', sep = '|')

#Taking out a row by a specific Service Date in dataframe
s_date_powerbi = pd.DataFrame()
s_date_powerbi = s_date_powerbi.append(powerbi.loc[powerbi['PayEndDTS'] == '2020-06-28 00:00:00.000'])

s_date_volume = pd.DataFrame()
day = [str(i) for i in range(15, 29)]
month = ['06']
year = ['2020']
for i in range(len(day)):
  date = month[0] + "/" + day[i] + "/" + year[0]
  s_date_volume = s_date_volume.append(qlik.loc[qlik['Service Date'] == date])


#Group by ChargedBusinessUnit, ChargeCD, and ChargedDepartmentID
temp_powerbi = s_date_powerbi.groupby(['ChargedBusinessUnit','ChargedDepartmentID','ChargeCD']).sum()
temp_volume = s_date_volume.drop_duplicates(subset=['Facility ID', 'Department Code' , 'Charge Code'])
temp_volume = temp_volume.groupby(['Facility ID', 'Department Code' , 'Charge Code']).sum()

temp_powerbi.drop(columns=['BindingID','UnitChargeAmountNBR','CalculatedVolumeNBR','WeightNBR','RevenueCD'], inplace=True)
temp_volume.drop(columns=['Unit Charge Amount', 'CombinedBillFLG'], inplace=True)

new_df = pd.concat([temp_powerbi, temp_volume], axis=1, join = 'inner')
new_df['new_column'] = new_df['ServiceUnitsNBR']-new_df['Service Units']

new_df = new_df[new_df['new_column'] != 0]
new_df.to_csv('validation2.csv')
