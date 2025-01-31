'''
Visualisation of the Feature Importance Analysis (data queried from DB)
'''

import altair as alt
import pandas as pd
import streamlit as st
import re

import config as cfg
from dbflow.src import db_utility as dbu

st.set_page_config(layout="wide")




# Function to convert CamelCase to spaced words and remove 'Classifier'
def format_model_name(name):
    # Insert spaces between words in CamelCase (except before the first word)
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    # Remove 'Classifier' from the string
    name = name.replace('Classifier', '')
    # Remove any trailing and leading whitespace
    return name.strip()



sql = f"""SELECT 

            importance_rank,  --- this is the criterium 
            feature, 
            ROUND(score,2) as score, 
            ROUND(score_abs,2) as score_abs,
            crop_type_code, 
            bbch_group,
            --jmd_treshold, 
            sample_limit,
            top_features,
            model,
            refpanel
            
        FROM
            uavfimportance
            
        WHERE
            --feature LIKE 'Mean%'
                (feature NOT LIKE '%DSM' OR feature NOT LIKE '%DSM%')
                AND feature NOT LIKE 'GLCM%' AND feature NOT LIKE 'GLDV%' AND feature NOT LIKE 'Radius%'
                AND feature NOT LIKE 'Elliptic%' AND feature NOT LIKE 'Roundness%' AND feature NOT LIKE 'Density%'
        ORDER BY
            crop_type_code, 
            bbch_group,
            model,
            score_abs DESC
            ;
        """


df = dbu.query_sql(sql, db_engine=cfg.dbarchive.archive.engine)

df['importance_rank'] += 1

# Apply the function to the 'model' column
df['model'] = df['model'].apply(format_model_name)

crop_list = df['crop_type_code'].unique().tolist()  # create list with all crop types
selected_crop = st.selectbox('select crop type:', options=crop_list)  # select crop type
selected_df_all = df.loc[df['crop_type_code'] == f'{selected_crop}']

if selected_crop == 'KM':
    R_Panel = 'RP'
else:
    R_Panel = 'RP'

#selected_df_all = df.loc[(df['crop_type_code'] == selected_crop) & (df['refpanel'] == R_Panel)]
selected_df_all = df.loc[(df['crop_type_code'] == selected_crop)]
#st.write(selected_df_all)


# ---record_number width - hight of the plot representation
# WW 200 # 600 - 600
# SG 230 # 600 - 800
# WR 130 # 600 - 900
# KM = 71 # 600 - 500

plot_param_dict = { # record_number width - hight
                     #'SG': [200, 200, 1380] #200
                     'SG': [300, 200, 850] #200
                    #,'WW': [210, 200, 1400]
                    ,'WW': [300, 200, 1250]
                    #,'WR': [120, 200, 1400]
                    ,'WR': [300, 200, 900]
                    #,'KM': [75, 200, 950]
                    ,'KM': [300, 200, 550]
                    }

record_number = plot_param_dict[selected_crop][0]



# Sort DataFrame by importance_rank and get top rows
selected_df = selected_df_all.sort_values(by='importance_rank').head(record_number)

# Remove 'Mean' from the values in the column for better representation
#selected_df['feature'] = selected_df['feature'].str.replace('Mean', '').str.strip()

dataframe = st.checkbox('Show Dataframe')
if dataframe:
    st.write(selected_df)


# Define chart configuration with font settings and explicit color settings for titles and labels
chart_config = {
    'axis': {
        'titleFont': 'Arial',
        'titleFontSize': 20,
        'titleColor': 'black',
        'labelFont': 'Arial',
        'labelFontSize': 14,
        'labelColor': 'black'
    }
}



highlight = alt.selection_point()
select = alt.selection_point(encodings=['x', 'y'])

legend_values = [1,2,3,4,5,6,7]

# Compute the counts for each feature
#counts = selected_df.groupby(['model', 'feature', 'importance_rank']).agg(total_count=('model', 'count')).reset_index()
counts = selected_df.groupby(['importance_rank']).agg(total_count=('model', 'count')).reset_index()

# Merge these counts back into the original DataFrame
#selected_df = selected_df.merge(counts, on=['model', 'feature', 'importance_rank'])
selected_df = selected_df.merge(counts, on=['importance_rank'])

# Sort the DataFrame by total_count in descending order before plotting
selected_df = selected_df.sort_values('total_count', ascending=False)
st.write(selected_df)

# Ensure the 'feature' column uses this order when plotting
selected_df['feature'] = pd.Categorical(selected_df['feature'], categories=selected_df['feature'].unique())


# Create the heatmap
heatmap = alt.Chart(selected_df).mark_rect().encode(
    x=alt.X('importance_rank:O', title='Importance Rank', axis=alt.Axis(labelLimit=1000, labelAngle=0)),
    y=alt.Y('feature:O', title='', axis=alt.Axis(labelLimit=1000),
            #sort=alt.EncodingSortField(field='count', op='max', order='descending'),
            sort=None
            ),
    color=alt.Color(
        'count(feature):Q',
        title=['Number ', 'of Occurrences'],
        scale=alt.Scale(scheme='greys'),
        legend=alt.Legend(format=',', labelColor='black', titleColor='black',
                         # values=legend_values
                        )
    ),
    tooltip=[
        alt.Tooltip('feature', title='Feature'),
        alt.Tooltip('importance_rank', title='Importance Rank'),
        alt.Tooltip('count(feature)', title='Number of Occurrences', format='d')
    ]
).add_params(
    highlight, select
).properties(
    width=plot_param_dict[selected_crop][1],
    height=plot_param_dict[selected_crop][2]
)



model_hist = alt.Chart(selected_df).mark_bar(color='lightgrey').encode(
    y=alt.Y('count(feature):Q', axis=alt.Axis(title='Count')),
    x=alt.X('model:N', axis=alt.Axis(labelLimit=1000), title='')
).transform_filter(select).properties(
#combined_chart = (heatmap).properties(
    width=100,
    height=200
)


# Combine charts
combined_chart = alt.hconcat(model_hist, heatmap
    ).configure_axis(**chart_config['axis']).configure_legend(
    titleFontSize=22,
    labelFontSize=24
)

# Apply chart configuration
combined_chart.configure(
    font='Helvetica',
    axis=alt.AxisConfig(labelFont='Helvetica', titleFont='Helvetica'),
    legend=alt.LegendConfig(labelFont='Helvetica', titleFont='Helvetica'),
    title=alt.TitleConfig(font='Helvetica')
)

model_hist.configure(
    font='Helvetica',
    axis=alt.AxisConfig(labelFont='Helvetica', titleFont='Helvetica'),
    legend=alt.LegendConfig(labelFont='Helvetica', titleFont='Helvetica'),
    title=alt.TitleConfig(font='Helvetica')
)

# Display the combined chart
st.write(combined_chart,  model_hist)
