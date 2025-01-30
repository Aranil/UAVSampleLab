'''
Visualisation of the distances after different features - for UAV Patterns
extracted from membership functions (eCognition)
'''
import streamlit as st
from pathlib import Path
import altair as alt


import config as cfg
from dbflow.src import db_utility as dbu


print(__file__)

logger = cfg.get_logger(logfilename=Path(__file__).name.replace('.py', '.log'))
logger.info(__file__)


FONT_SIZE = 18

# Define chart configuration first
chart_config = {
    'axis': {
        'titleFont': 'Arial',
        'titleFontSize': FONT_SIZE-4,
        'titleColor': 'black',
        'labelFont': 'Arial',
        'labelFontSize': FONT_SIZE -4,
        'labelColor': 'black'
    },
    'legend': {
        'titleFontSize': FONT_SIZE -4 ,
        'labelFontSize': FONT_SIZE -4,
        'labelFont': 'Helvetica',
        'titleFont': 'Helvetica',
        'titleColor': 'black',
        'labelColor': 'black'
    },
    'title': {
        'font': 'Helvetica',
        'color': 'black',
        'fontSize': FONT_SIZE
    }
}



col01, col02 = st.columns((1, 1))
with col01:
    CROP_TYPE = st.selectbox(label='Select AOI:', options=['WR', 'WW', 'SG', 'WG', 'ZwFr', 'KM', 'SM']) #, 'WG', 'WR', 'ZwFr', 'KM'
with col02:
    THRESHOLD = st.slider(label='Overlap threshold:', min_value=0.0, max_value=1.0, value=0.30, step=0.01)

st.write('')


sql_filter = f"""
                SELECT 
                    crop_type_code,
                    feature,
                    layer, 
                    overlap, 
                    class1,
                    class2
                FROM 
                    uavmembershipf
                WHERE 
                    overlap <= {THRESHOLD}
                AND
                    overlap NOT NULL
                AND 
                    crop_type_code == '{CROP_TYPE}'
                GROUP BY 
                
                    class1, 
                    class2,
                    layer
                ORDER BY 
                    layer,
                    class1, 
                    class2,
                    feature
                ;
            """
source = dbu.query_sql(sql_filter, db_engine=cfg.dbarchive.archive.engine)

print(source['class1'])
source['class1'] = source['class1'].str.replace("_", " ")
source['class2'] = source['class2'].str.replace("_", " ")

st.write(source)

col21, col22 = st.columns((1, 4))

highlight = alt.selection_multi(fields=['count'], bind='legend')
click = alt.selection_multi(encodings=['x', 'y'], bind='legend')

# Configure common options
rect = alt.Chart(source
).transform_aggregate(
    count='count():O',
    groupby=['class1', 'class2', 'layer']
).encode(
    alt.X('class1:O', scale=alt.Scale(paddingInner=0), title='Class 1', axis=alt.Axis(labelLimit=150)),
    alt.Y('class2:O', scale=alt.Scale(paddingInner=0), title='Class 2', axis=alt.Axis(labelLimit=150)),
    #color=alt.condition(highlight, 'count()', alt.value('lightgray')),
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    tooltip=['class1', 'class2', 'count()']
).mark_rect(stroke='lightgrey', strokeWidth=1
).encode(
        color=alt.Color('count(overlap):O', scale=alt.Scale(scheme='greenblue')),
        #order=alt.Order('count(jmd):O', sort='ascending')
        #legend=alt.Legend(direction='horizontal'))
).properties(
    width=170,
    height=170
).add_selection(
    highlight,
 #   selector,
    click
)



# ---- PLot Heat plots
heatmap3 = alt.Chart(source
).mark_rect(stroke='lightgrey', strokeWidth=1).encode(
    x=alt.X('layer:O', sort=alt.EncodingSortField(field='layer', op='count', order='descending'), title='Layer'),
    y=alt.Y('feature:O', sort=alt.EncodingSortField(field='feature', op='count', order='ascending'), title='Feature'),
    color=alt.Color('count(overlap):Q', scale=alt.Scale(scheme='greenblue')), # scheme='redblue',
    #order=alt.Order('count(jmd):Q', sort='descending'),
    #opacity=alt.condition('count(jmd):Q' < selector.cutoff, alt.OpacityValue(1), alt.OpacityValue(0.1)),
    tooltip=['feature', 'layer',  'count(overlap)', 'layer:N', 'feature']
).transform_filter(
    click,
).properties(
    width=600,
    height=50 * len(source['feature'].unique())
).add_selection(
    highlight
)


# Concatenate the charts
plot = alt.hconcat(heatmap3, rect).configure_axis(
    **chart_config['axis']
).configure_legend(
    **chart_config['legend']
).configure_title(
    **chart_config['title']
)


st.write(plot)
