import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import numpy as np
import plotly

app = dash.Dash(sharing=True , csrf_protect = False)
server = app.server


#app.scripts.config.serve_locally = True
# app.css.config.serve_locally = True


#this is where you upload the files and also apply the sheet name


xls = pd.ExcelFile('C:\\Users\\hp\\Anaconda3\\data science course edx\\Copy of 30052018_ELSS Research.xlsx')
elss = pd.read_excel(xls , 'sheet1')


#inputting the categories

#cat = input("enter the categories name  ")
status_distri = input("enter the distribution status - Accumulated")
purchase = input("enter the mode of purchase - 1 or 2")

#status_distri = 'Accumulated'
#purchase = '1'

#elss = elss[(elss['CategoryName']==cat)&(elss['DistributionStatus']==status_distri) &(elss['PurchaseMode']==purchase)]
elss = elss[(elss['DistributionStatus']==status_distri) &(elss['PurchaseMode']==purchase)]

#print(elss)

#elss.count()

n = min(elss['Year5'].count() , elss['Year1'].count() ,elss['Year2'].count() ,elss['Year3'].count() ,elss['Year4'].count())
#elss.count()
#elss['Year1'][0]
#print(n)

#type(elss['AsOfOriginalReported'].iloc[2])
elss['AsOfOriginalReported'] = pd.to_numeric(elss['AsOfOriginalReported'], errors='coerce')
elss['AsOfOriginalReported'] = elss['AsOfOriginalReported']/10000000
#elss[elss['AsOfOriginalReported']/100000 > 1300]
#elss

#type(elss['AsOfOriginalReported'].iloc[2])
#print(elss['AsOfOriginalReported'])

#print(elss[elss['AsOfOriginalReported'] > 1000])


df = elss[elss['AsOfOriginalReported'] > 1300]
#print(df)

df['EquitySectorBasicMaterialsLongRescaled'] = pd.to_numeric(df['EquitySectorBasicMaterialsLongRescaled'], errors='coerce')
df['EquitySectorCommunicationServicesLongRescaled'] = pd.to_numeric(df['EquitySectorCommunicationServicesLongRescaled'], errors='coerce')
df['EquitySectorConsumerCyclicalLongRescaled'] = pd.to_numeric(df['EquitySectorConsumerCyclicalLongRescaled'], errors='coerce')
df['EquitySectorConsumerDefensiveLongRescaled'] = pd.to_numeric(df['EquitySectorConsumerDefensiveLongRescaled'], errors='coerce')
df['EquitySectorEnergyLongRescaled'] = pd.to_numeric(df['EquitySectorEnergyLongRescaled'], errors='coerce')
df['EquitySectorFinancialServicesLongRescaled'] = pd.to_numeric(df['EquitySectorFinancialServicesLongRescaled'], errors='coerce')
df['EquitySectorHealthcareLongRescaled'] = pd.to_numeric(df['EquitySectorHealthcareLongRescaled'], errors='coerce')
df['EquitySectorIndustrialsLongRescaled'] = pd.to_numeric(df['EquitySectorIndustrialsLongRescaled'], errors='coerce')
df['EquitySectorRealEstateLongRescaled'] = pd.to_numeric(df['EquitySectorRealEstateLongRescaled'], errors='coerce')
df['EquitySectorTechnologyLongRescaled'] = pd.to_numeric(df['EquitySectorTechnologyLongRescaled'], errors='coerce')
df['EquitySectorUtilitiesLongRescaled'] = pd.to_numeric(df['EquitySectorUtilitiesLongRescaled'], errors='coerce')


#print("enter the sectoral filter value")
filter_value = input("enter the sectoral filter value")
#filter_value = '40'

df = df.drop(df[df.EquitySectorBasicMaterialsLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorCommunicationServicesLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorConsumerCyclicalLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorConsumerDefensiveLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorEnergyLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorFinancialServicesLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorHealthcareLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorIndustrialsLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorRealEstateLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorTechnologyLongRescaled > int(filter_value)].index)
df = df.drop(df[df.EquitySectorUtilitiesLongRescaled > int(filter_value)].index)

df['Year1'] = pd.to_numeric(df['Year1'] , errors='coerce')
df['Year2'] = pd.to_numeric(df['Year2'], errors='coerce')
df['Year3'] = pd.to_numeric(df['Year3'], errors='coerce')
df['Year4'] = pd.to_numeric(df['Year4'], errors='coerce')
df['Year5'] = pd.to_numeric(df['Year5'], errors='coerce')

#print(df[['Year1' , 'Year2' , 'Year3' , 'Year4' , 'Year5']])
#(np.isnan(elss['Year5'].iloc[i])) == np.False_ & (np.isnan(elss['Year4'].iloc[i])) == np.False_ & (np.isnan(elss['Year3'].iloc[i])) == np.False_


#type(df['Year4'].iloc[3])
#type(elss['Year4'].iloc[3])
df = df.replace(' ' , np.NaN)
pd.isnull(df['Year4'].iloc[3])
#np.True_

#df['Year5']
#print(df['LegalName'])
#(pd.isnull(elss['Year5'].iloc[i])) == False and (pd.isnull(elss['Year4'].iloc[i])) == np.False_ and (pd.isnull(elss['Year3'].iloc[i])) == False
#np.isnan(np.array([np.nan, 0], dtype=np.float64))

#legal_name = input("enter the column to be analysed")
#df[legal_name]
#print(df['Year4'])

weighted_average_return = []

for i in range(0, len(df['ISIN'])):
    x = 0

    if (pd.isnull(df['Year5'].iloc[i])) == False and (pd.isnull(df['Year4'].iloc[i])) == False and (pd.isnull(df['Year3'].iloc[i])) == False:
        x = x + 0.3 * df['Year5'].iloc[i] + 0.25 * df['Year4'].iloc[i] + 0.2 * df['Year3'].iloc[i] + 0.15 * df['Year2'].iloc[i] + 0.1 * df['Year1'].iloc[i]
        weighted_average_return.append(x)
        #print(1)

    if (pd.isnull(df['Year5'].iloc[i])) == True and (pd.isnull(df['Year4'].iloc[i])) == False and (pd.isnull(df['Year3'].iloc[i])) == False:
        x = x + 0.35 * df['Year4'].iloc[i] + 0.3 * df['Year3'].iloc[i] + 0.2 * df['Year2'].iloc[i] + 0.15 * df['Year1'].iloc[i]  # + 0.1*elss['Year1'][i]
        weighted_average_return.append(x)
        #print(2)

    if (pd.isnull(df['Year5'].iloc[i])) == True and (pd.isnull(df['Year4'].iloc[i])) == True and (pd.isnull(df['Year3'].iloc[i])) == False:
        x = x + 0.4 * df['Year3'].iloc[i] + 0.35 * df['Year2'].iloc[i] + 0.25 * df['Year1'].iloc[i]  # + 0.15*elss['Year2'][i] + 0.1*elss['Year1'][i]
        weighted_average_return.append(x)
        #print(3)

    if (pd.isnull(df['Year5'].iloc[i])) == True and (pd.isnull(df['Year4'].iloc[i])) == True and (pd.isnull(df['Year3'].iloc[i])) == True:
        x = x + 0.6 * df['Year2'].iloc[i] + 0.4 * df['Year1'].iloc[i]  # + 0.2*elss['Year3'][i] + 0.15*elss['Year2'][i] + 0.1*elss['Year1'][i]
        weighted_average_return.append(x)
        #print(4)

#print(weighted_average_return)

df['SharpeRatio3Yr'] = pd.to_numeric(df['SharpeRatio3Yr'])
df['SortinoRatio3Yr'] = pd.to_numeric(df['SortinoRatio3Yr'])
df['TreynorRatio3Yr'] = pd.to_numeric(df['TreynorRatio3Yr'])
df['InformationRatio3Yr'] = pd.to_numeric(df['InformationRatio3Yr'])
df['Alpha3Yr'] = pd.to_numeric(df['Alpha3Yr'])

#print(df['SharpeRatio3Yr'].count())

rat = []

for i in range(0, len(df['ISIN'])):
    x = 0
    x = x + (df['SharpeRatio3Yr'].iloc[i] + df['SortinoRatio3Yr'].iloc[i] + df['TreynorRatio3Yr'].iloc[i] +
             df['InformationRatio3Yr'].iloc[i] + df['Alpha3Yr'].iloc[i]) / 5

    rat.append(x)

#print(rat , weighted_average_return)

df['e'] = pd.Series(weighted_average_return , index = df.index)
df['f'] = pd.Series(rat, index=df.index)

df['weight_ranked'] = df['e'].rank(ascending=0,method='dense')
df['rat_ranked']=df['f'].rank(ascending=0,method='dense')

average_rank = []

for i in range(0,len(df['ISIN'])):
    x=0
    x = x + df['rat_ranked'].iloc[i] + df['weight_ranked'].iloc[i]
    average_rank.append(x)

df['final'] = pd.Series(average_rank , index = df.index)

df['yoo']=df['final'].rank(ascending=1,method='dense')

#print(df)

DF_WALMART = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/1962_2006_walmart_store_openings.csv')

DF_GAPMINDER = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv'
)
DF_GAPMINDER = DF_GAPMINDER[DF_GAPMINDER['year'] == 2007]
DF_GAPMINDER.loc[0:20]

DF_SIMPLE = pd.DataFrame({
    'x': ['A', 'B', 'C', 'D', 'E', 'F'],
    'y': [4, 3, 1, 2, 3, 6],
    'z': ['a', 'b', 'c', 'a', 'b', 'c']
})

ROWS = [
    {'a': 'AA', 'b': 1},
    {'a': 'AB', 'b': 2},
    {'a': 'BB', 'b': 3},
    {'a': 'BC', 'b': 4},
    {'a': 'CC', 'b': 5},
    {'a': 'CD', 'b': 6}
]


app.layout = html.Div([
    html.H4('Mutual Fund Ranking Table'),
    dt.DataTable(
        rows=df.to_dict('records'),  #DF_GAPMINDER

        # optional - sets the order of columns
        columns=sorted(df.columns),   #DF_GAPMINDER

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-gapminder'
    ),
    html.Div(id='selected-indexes'),
    dcc.Graph(
        id='graph-gapminder'
    ),
], className="container")


@app.callback(
    Output('datatable-gapminder', 'selected_row_indices'),
    [Input('graph-gapminder', 'clickData')],
    [State('datatable-gapminder', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):

    print("CLICKDATA IS" , clickData)

    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices

#print(clickdata)

#def update_rank():

#graph update snippet->

@app.callback(
    Output('graph-gapminder', 'figure'),
    [Input('datatable-gapminder', 'rows'),
     Input('datatable-gapminder', 'selected_row_indices')])
def update_figure(rows, selected_row_indices):
    dff = pd.DataFrame(rows)
    fig = plotly.tools.make_subplots(
        rows=3, cols=1,
        subplot_titles=('Ranking', 'AUM', 'ISIN',),  #life expectancy -> yoo , GDP-> AsOfOriginalReported , #Population -> ISIN
        shared_xaxes=True)
    marker = {'color': ['#0074D9']*len(dff)}
    for i in (selected_row_indices or []):
        marker['color'][i] = '#FF851B'
    fig.append_trace({
        'x': dff['LegalName'],
        'y': dff['yoo'],
        'type': 'bar',
        'marker': marker
    }, 1, 1)
    fig.append_trace({
        'x': dff['LegalName'],
        'y': dff['AsOfOriginalReported'],
        'type': 'bar',
        'marker': marker
    }, 2, 1)
    fig.append_trace({
        'x': dff['LegalName'],
        'y': dff['ISIN'],
        'type': 'bar',
        'marker': marker
    }, 3, 1)
    fig['layout']['showlegend'] = False
    fig['layout']['height'] = 800
    fig['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 60,
        'b': 200
    }
    fig['layout']['yaxis3']['type'] = 'log'
    return fig


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)# , port=8000)