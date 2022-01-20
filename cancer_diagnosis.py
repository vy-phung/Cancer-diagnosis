# Import required libraries
from ast import Return
from http import server
from cgitb import text
from click import style
from gc import callbacks
from turtle import width
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, callback_context 
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter 
import itertools
# Read data into pandas dataframe
cancer = pd.read_csv("data.csv")
x = cancer.iloc[:,2:32].columns.values.tolist()
y = cancer.iloc[:,2:32].columns.values.tolist()
# Create a dash application
app = dash.Dash(__name__)
server = app.server
# Create an app layout
app.layout = html.Div(children=[html.H1('Breast Cancer diagnosis',
                                        style={'textAlign': 'center', 'color': 'black', 'font-family':'Arial',
                                               'font-size': 40,}),
                                html.Hr(),
                                #table               
                                html.Div([
                                html.H3('First 5 rows and 10 columns in cancer dataset',style={'textAlign':'left','color':'black',
                                'font-size':30,'font-family':'Arial'}),
                                dash_table.DataTable(id='table',
                                columns = [{'name':i,'id':i} for i in cancer.iloc[:,2:12]],
                                data=cancer.iloc[:,2:12].head().to_dict('records'),
                                )],style = {'backgroundColor':'white','padding':'1rem',
                                    'boxShadow': '#e3e3e3 4px 4px 2px'}),
                                html.Br(),
                                # Visualization                              
                                html.Div([
                                html.H3('Data visualization',
                                        style={'textAlign': 'left', 'color': 'black',
                                               'font-size': 30,}),               
                                dcc.Dropdown(id='x',
                                options = [{'label': i, 'value': i} for i in x
                                ],                               
                                value = 'radius_mean',
                                placeholder = 'Place holder here',
                                searchable = True,style={'width':'50%','display':'inline-block'}),
                                dcc.Dropdown(id='y',
                                options = [{'label': i, 'value': i} for i in y
                                ],                               
                                value = 'concave points_mean',
                                placeholder = 'Place holder here',
                                searchable = True,style={'width':'50%','display':'inline-block'}),
                                dcc.Graph(id='scatter-plot')
                                ],style={'font-family':'Arial','backgroundColor':'rgb(81, 131, 310)','padding':'1rem',
                                'boxShadow': '#e3e3e3 4px 4px 2px'}),
                                html.Br(),
                                # predict
                                html.Div([
                                    html.H3('Predict radius mean by Linear regression',
                                            style={'textAlign': 'left', 'color': 'black',
                                                'font-size': 30,}),
                                    html.P('You have to type number for 3 columns below in order to predict the radius mean',
                                            style={'textAlign': 'left', 'color': 'black',
                                                'font-size':25}),
                                    html.Label('Perimeter'), html.Br(),
                                    dcc.Input(id='perimeter',type='number',min=30,max=200, step = 0.01,
                                    placeholder='perimeter_mean range from 30-200', value=122.80,
                                    style={'width':'32%',}), html.Br(),
                                    html.Label('Area'), html.Br(),
                                    dcc.Input(id='area',type='number',min=130,max=3000, step = 0.01,
                                    placeholder='area_mean range from 130-3000', value = 1001.00,
                                    style={'width':'32%',}), html.Br(),
                                    html.Label('Concave points'), html.Br(),
                                    dcc.Input(id='concave',type='number',min=0,max=1, step = 0.00001,
                                    placeholder='concave points_mean range from 0-1', value = 0.14710,
                                    style={'width':'32%',}),

                                    html.Br(),
                                    html.Br(),
                                    html.Div(id='predict',style={ 'font-size':20,'color':'black','textAlign':'left', 
                                          }),
                                    ],
                                    style={'font-family':'Arial','backgroundColor':'rgb(245, 245, 285)','padding':'1rem',
                                    'boxShadow': '#e3e3e3 4px 4px 2px'}),
                                    html.Br(),
                                    # confusion
                                    html.Div([
                                    html.H3('SVM Classification and confusion matrix',
                                            style={'textAlign': 'left', 'color': 'black',
                                                'font-size': 25,}),
                                    html.P('Type number of test size and variables to train SVM model. After choosing, click get figure and wait for minutes',
                                            style={'textAlign': 'left', 'color': 'black',
                                                'font-size':23}),          
                                    html.Label('Test size'), html.Br(),
                                    dcc.Input(id='Test-size',type='number',min=0,max=1, step = 0.1,
                                    placeholder='Test size', value=0.2,
                                    style={'width':'35%',}), html.Br(),
                                    html.Label('Select variables'), html.Br(),
                                    dcc.Dropdown(id='variables',
                                    options = [{'label': i, 'value': i} for i in x],
                                    placeholder='Choose variables',
                                    value = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean'],
                                    multi = True,
                                    searchable = True,
                                    style={'width':'100%',}),
                                    html.Br(),
                                    html.Br(),
                                    html.Div(id='classify',style={ 'font-size':20,'color':'black','textAlign':'left', 
                                          }),
                                    html.Button('Get figure', id = 'figure')           
                                    ],
                                    style={'font-family':'Arial','backgroundColor':'whitesmoke','padding':'1rem',
                                    'boxShadow': '#e3e3e3 4px 4px 2px'}),
                                    ])
# scatter plot
@app.callback( Output(component_id='scatter-plot', component_property='figure'),
               Input(component_id='x', component_property='value'),
               Input(component_id='y',component_property='value'))
def scatter_plot(x,y): 
    fig = px.scatter(cancer,x=str(x),y=str(y), color='diagnosis') 
    return fig
# Predict
@app.callback( Output('predict','children'),
               Input('perimeter','value'),
               Input('area','value'),
               Input('concave','value'))
def predict_radius(perimeter,area,concave):
    df = pd.DataFrame({'perimeter_mean':[perimeter],'area_mean':[area],'concave points_mean':[concave]})
    lr = LinearRegression()
    x= cancer[['perimeter_mean','area_mean','concave points_mean']]
    y= cancer[['radius_mean']]
    lr.fit(x,y)
    ypred = lr.predict(df)
    return 'Predicted radius mean {}'.format(ypred[0,0])
# Training model     
@app.callback( Output('classify','children'),
                Input('Test-size','value'),
                Input('variables','value'),
                Input('figure','n_clicks'))
def classify(test_size,var,click):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if click is None:
        raise PreventUpdate 
    else:        
    # test-train split
        y = cancer['diagnosis'].values
        x = cancer[var].values
        x = preprocessing.StandardScaler().fit(x).transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=4)
        parameters_svm = {'kernel':('linear','rbf','poly','rbf','sigmoid'),
                    'C': np.logspace(-3, 3, 5),
                    'gamma':np.logspace(-3, 3, 5)}
        svm = SVC()
        svm_cv = GridSearchCV(svm,parameters_svm,cv=10)
        svm_cv.fit(x_train, y_train)
        sv_cv = svm_cv.predict(x_test)
        # plot 
        plt.figure()
        cm = confusion_matrix(y_test,sv_cv)
        classes=['Benign(B)','Malignant(M)']
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix for SVM')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=360)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')  
        fig = plt.show() 
        return fig, click==None    
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
