import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle
import numpy as np
app = dash.Dash()

def load_model():
    file_name = "model_XGBC.p"
    with open(file_name, "rb") as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model
def Load_Dictionary():
    with open('Dicts\Cities.pkl', 'rb') as f:
        cities = pickle.load(f)

    with open('Dicts\States.pkl', 'rb') as f:
        states = pickle.load(f)
        
    with open('Dicts\Merchants.pkl', 'rb') as f:
        merchants = pickle.load(f)

    with open('Dicts\Jobs.pkl', 'rb') as f:
        jobs = pickle.load(f)
        
    with open('Dicts\Days.pkl', 'rb') as f:
        days = pickle.load(f)

    with open('Dicts\Category.pkl', 'rb') as f:
        category = pickle.load(f)
    return cities, states, merchants, jobs, days, category
def Converting_Dictionaries():
    dicts = Load_Dictionary()
    LIST = ["cities", "states", "merchants", "jobs", "days", "category"]
    HTML_DICT = {}
    for i, dict in enumerate(dicts):
        temp_dict = {}
        temp_list = []
        for key, value in enumerate(dict):
            temp_dict = {"label" : value, 'value' : key}
            temp_list.append(temp_dict)
        HTML_DICT[LIST[i]] = temp_list
    return HTML_DICT
HTML_DICT = Converting_Dictionaries()
print(HTML_DICT["days"])
app.layout = html.Div( children=
    [ 
        html.H1(children='Predict Fraud',
                style={
                    'textAlign' : 'center'
                }),
        html.Br(),
        html.Br(),
        #Taking Input for model
        html.H4(children='Hour - '),
        html.Br(),
        

        dcc.Input(
            id = 'Hour',
            placeholder = "Hour",
            type = 'text', 
            value = ''
            ),
        html.Br(),
        html.Br(),

        dcc.Input(
            id = 'Age',
            placeholder = "Age",
            type = 'text', 
            value = ''
            ),
        html.Br(),
        html.Br(),

        dcc.Input(
            id = 'Fraud_avg',
            placeholder = "Fraud Amt",
            type = 'text', 
            value = '',
            n_submit = 1,
           ),
        html.Br(),
        html.Br(),

        dcc.Input(
            id = 'Amt_Avg_Legit',
            placeholder = "Legit Amount",
            type = 'text', 
            value = ''
            ),
        html.Br(),
        html.Br(),

        dcc.Input(
            id = 'Total_Amt',
            placeholder = "Total Amount",
            type = 'text', 
            value = ''
            ),
        html.Br(),
        html.Br(),

        dcc.Input(
            id = 'Distance',
            placeholder = "Distance between Merchant and Buyer",
            type = 'text', 
            value = ''
            ),
        html.Br(),
        html.Br(),
         
        dcc.Dropdown(
            id = 'Day',
            options = HTML_DICT["days"],
            placeholder = "Select Day",
            value = ""
             ),
        html.Br(),
        html.Br(),

        dcc.Dropdown(
            id = 'State',
            options = HTML_DICT['states'],
            placeholder = "Select State",
            value = ""
             ),
        html.Br(),
        html.Br(),

        dcc.Dropdown(
            id = 'Merchant',
            options = HTML_DICT['merchants'],
            placeholder = "Select Merchant",
            value = ""
             ),
        html.Br(),
        html.Br(),

        dcc.Dropdown(
            id = 'Job',
            options = HTML_DICT['jobs'],
            placeholder = "Select Job",
            value = ""
             ),
        html.Br(),
        html.Br(),

        dcc.Dropdown(
            id = 'Category',
            options = HTML_DICT['category'],
            value = "",
            placeholder = "Select Category",
             ),
        html.Br(),
        html.Br(),
        

        html.Button("Submit", id = "submit-form"),

        html.Br(),
        html.Br(),
        html.Div(id = "Output")
    ]
)


@app.callback(
    Output(component_id='Output', component_property='children'),
    [Input(component_id="Hour", component_property="value")],
    [Input(component_id="Age", component_property="value")],
     [Input(component_id="Fraud_avg", component_property="value")],
     [Input(component_id="Amt_Avg_Legit", component_property="value")],
     [Input(component_id="Total_Amt", component_property="value")],
    [Input(component_id="Distance", component_property="value")],
    [Input(component_id="Day", component_property="value")],
    [Input(component_id="State", component_property="value")],
    [Input(component_id="Merchant", component_property="value")],
    [Input(component_id="Job", component_property="value")],
    [Input(component_id="Category", component_property="value")]   
)
    
def Get_prediction(Hour, Age, Frauad_Avg, Amt_Avg_Legit, Total_Amt, Distance, Day, State, Merchant, Job, Category):
    #print(Hour, Age, Frauad_Avg, Amt_Avg_Legit, Total_Amt, Distance, Day, State, Merchant, Job, Category)
    #print(np.size(np.array([Hour ,Age ,Frauad_Avg, Amt_Avg_Legit, Total_Amt, Distance, Day, State, Merchant, Job, Category,Frauad_Avg]).reshape(-1, 1).astype(np.float)))
    model = load_model()
    Data = np.array([Hour ,State ,Merchant, Age, Day
                    , Job, Category, Total_Amt, Distance, Frauad_Avg, Amt_Avg_Legit,Frauad_Avg]).reshape(-1, 1).astype(np.float).transpose()
    preds = model.predict(Data).tolist()
    #print(preds)
    if preds[0] == 0:
        return "Transaction is seems to be legit"
    else:
        return "Transaction is Fraud" 

if __name__ == "__main__":
    app.run_server(port = 4050)