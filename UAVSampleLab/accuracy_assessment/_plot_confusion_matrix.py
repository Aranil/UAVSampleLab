'''
To plot the Confusion Matrix for each Crop Type Model
Reads Accuracy Metrics from DB table -> 'uavcnnaccuracy'
'''

import pathlib
import pandas as pd
import numpy as np

import config as cfg
from dbflow.src import db_utility as dbu
from accuracy_assessment import accuracy_utils as acc

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'




train_model = {'yes': 'Used for Training of CNN'
                , 'no': 'Not Used for Training of CNN'
                }

calulate_accuracy_metrics = True
insert2db = True
normalize_confusion_matrix = True
even_out_labels = False

sample_used_in_cnn = 'no'

crop_type_list = ['WW']
#crop_type_list = ['WR']
#crop_type_list = ['KM']
#crop_type_list = ['SG']
PATTERN = 'mosmethod9'





if calulate_accuracy_metrics:

    dbarchive = dbu.connect2db(cfg.db_path)

    sql = f"""
            SELECT 

                label_actual, 
                label_predicted, 
                sum(user_class) as user_class, 
                sl_nr, 
                crop_type_code, 
                pattern, 
                unit, 
                sample_used_in_cnn

            FROM 
                uavcnnaccuracy
            WHERE
            	pattern=='xx'
            --	user_class != 0
            --AND unit== 'reflectance'
            --AND unit== 'DN'
            --AND label_actual=='bare_soil'
            --AND label_predicted=='bare_soil'
            --AND 
            --pattern NOT LIKE 'mosmethod%'

            --AND unit == 'reflectance'
            --AND sample_used_in_cnn == 'no'
            --AND crop_type_code == 'WR'
            --AND pattern == 'Blure'

           -- WHERE 
           -- pattern LIKE 'mosmethod%'
            --sl_nr == '190-00'

            GROUP BY 
                label_actual, 
                label_predicted,
                --sl_nr, 
                crop_type_code, 
                pattern, 
                unit, 
                sample_used_in_cnn

            ORDER BY
                unit,
                pattern,
                label_actual,
                label_predicted,
                pattern
            ;

    """

    df_query = pd.read_sql(sql, dbarchive.archive.engine)

    num_rows = 1
    #num_cols = len(df_query.groupby(['crop_type_code', 'pattern', 'unit', 'sample_used_in_cnn']))

    if sample_used_in_cnn == 'yes':
        if len(crop_type_list) > 1:
            num_cols = len(crop_type_list)
        else:
            num_cols = 2
    elif sample_used_in_cnn == 'no':
        if len(crop_type_list) > 1:
            num_cols = len(crop_type_list)
        else:
            num_cols = 2



    # Create a single facet with multiple subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, 6))

    df_query = df_query.query("sample_used_in_cnn == @sample_used_in_cnn and crop_type_code in @crop_type_list")
    # group by the data to be plotted after
    for j, (i, df) in enumerate(df_query.groupby(['crop_type_code', 'pattern', 'unit', 'sample_used_in_cnn'])):

        dff = \
        df.groupby(['crop_type_code', 'pattern', 'unit', 'sample_used_in_cnn', 'label_actual', 'label_predicted'])[
            'user_class'].sum().reset_index()
        df_main = dff[['label_actual', 'label_predicted', 'user_class']]

        # Use the replace method to change values in column 'B' based on the dictionary
        df_main['label_actual'] = df_main['label_actual'].replace(acc.tick_labels)
        df_main['label_predicted'] = df_main['label_predicted'].replace(acc.tick_labels)



        if even_out_labels == True:
            # Check if values from the list exist in the DataFrame and add rows for missing values
            missing_values_a = [value for value in acc.tick_labels.values() if value not in df_main['label_actual'].values]
            missing_values_p = [value for value in acc.tick_labels.values() if value not in df_main['label_predicted'].values]

            print(missing_values_a)
            print(missing_values_p)

            # Create a new empty DataFrame with the same columns as df
            empty_df = pd.DataFrame(columns=df_main.columns)

            for a in missing_values_a:
                for p in missing_values_p:
                    # Create a dictionary for the new row with the value for 'Column1'
                    new_row = {'label_actual': a, 'label_predicted': p}
                    # Append the new row to the empty DataFrame
                    empty_df = empty_df.append(new_row, ignore_index=True)
            # Use drop_duplicates to find unique combinations of 'Column1' and 'Column2'
            empty_df = empty_df[['label_actual', 'label_predicted']].drop_duplicates()
            print(empty_df)

            df_main_ = df_main

            # Concatenate the original DataFrame and the new rows
            df_main = pd.concat([df_main_, empty_df], ignore_index=True)



        # mask is not used!

        # Create a mask where both 'label_actual' and 'label_predicted' are 0
        #mask = (df_main['label_actual'] == 0) & (df_main['label_predicted'] == 0)

        # Set values to NaN where the mask is True
        #df_main.loc[mask, ['label_actual', 'label_predicted']] = None


        # print out columns
        # print(df['sl_nr'].unique())
        crop_type_code = dff['crop_type_code'].unique()[0]


        print(dff['pattern'].unique())

        unit = dff['unit'].unique()[0]

        sample_used_in_cnn = dff['sample_used_in_cnn'].unique()[0]
        print(dff)


        outdir = pathlib.Path(cfg.output_path).joinpath(r'_plots\_accuracy')
        outdir.mkdir(parents=True, exist_ok=True)

        OUTFILE = outdir.joinpath("confusion_matrix_{}_{}_{}.png".format(sample_used_in_cnn, unit, crop_type_list))
        print(OUTFILE)



        # select labels with non zero values at user_class and actual class
        non_zero_labels = df_main.query("user_class!=0")
        labels_ = list(pd.unique(non_zero_labels[['label_actual', 'label_predicted']].values.ravel()))
        selected_labels = df_main.query("label_actual == @labels_ & label_predicted == @labels_")

        # convert back to origin form
        dfff = selected_labels.pivot(index='label_predicted', values='user_class', columns='label_actual')

        # print(dff.index)
        # print(dff.columns)



        print('+' * 20)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # print(dff.reset_index().iloc[:, :])

        # this is confusion matrix for all classes
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            # print(dfff.reset_index().iloc[:, 1:])
            print(dfff)

        # extract matrix from df
        # df_subset = df.loc[df['User Class \ Sample'].isin(classes)]
        # print(df_subset)

        # confusion matrix
        # A = df_subset.iloc[:, 1:-2].astype(int).to_numpy()
        #mask_ = dfff == None

        dfff = dfff.replace(np.nan, 0)
        class_names = dfff.columns
        A = dfff.reset_index().iloc[:, 1:].astype(int).to_numpy()
        print(A)  # confusion matrix for all classes together

        if normalize_confusion_matrix == True:
            # normalize the confusion matrix
            normalized_A = A.astype('float') / A.sum(axis=1)[:, np.newaxis]
            print(normalized_A)

            # convert nan to 0, round value, convert to int
            A = np.round(np.nan_to_num(normalized_A * 100), 0).astype(int)
            normalized_A = None

        print(normalized_A)

        # ----  calculate number of True Positive (TP) & True Negative (TN) samples
        TP, FP, FN, TN, TotalSamp = acc.classification_report(A, classes=class_names)
        print('-' * 80)

        print(dff['crop_type_code'].unique())

        # Instantiate the confusion matrix DataFrame with index and columns
        # dtype=int is important for Windows users
        dfA = pd.DataFrame(A, index=class_names, columns=class_names, dtype=int)
        print(dfA)


        # if plot_accuracy_matrix == True:


        # Create a heatmap using Seaborn
        #plt.figure(figsize=(8, 6))
        # Loop through the data list and plot each item in a subplot
        mask_ = dfA == np.nan

        ax = axes[j]

        # Define a custom colormap transitioning from white to green
        colors = [(1, 1, 1), (0.6, 0.8, 0.6), (0, 0.5, 0)]  # White to Green
        cmap = LinearSegmentedColormap.from_list('WhiteToGreen', colors, N=256)

        sns.set(font_scale=1.2)
        sns.heatmap(dfA, annot=True,
                    fmt="d",
                    #fmt=".1f",
                    mask=mask_,
                    cmap=cmap,
                    cbar=False,
                    square=True,
                    linewidths=0.5,
                    ax=ax
                    )

        # Add labels and title
        #ax.set_xlabel("Predicted", labelpad=20)
        #ax.set_ylabel("True", labelpad=20)
        ax.set_xlabel(" ", labelpad=25)
        ax.set_ylabel(" ", labelpad=25)
        #ax.set_title("{} [{}] - {}".format(cfg.crop_types[crop_type_code], unit, train_model[sample_used_in_cnn]))
        ax.set_title("{} [{}] ".format(cfg.crop_types[crop_type_code], unit), y=1.1, fontsize=22)

        # Set the font size for x and y labels
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)


        # Set x and y label titles for the entire facet
        fig.text(0.07, 0.6, "True", va="center", rotation="vertical", fontsize=20) # y
        #fig.text(0.5, 0.00, "Predicted", ha="center", fontsize=20) # x
        if sample_used_in_cnn == 'yes':
            if len(crop_type_list) == 1:
                fig.text(0.35, 0.00, "Predicted", ha="center", fontsize=20)
            elif len(crop_type_list) == 2:
                fig.text(0.5, 0.00, "Predicted", ha="center", fontsize=20) # x

        elif sample_used_in_cnn == 'no':
            #fig.text(0.4, 0.00, "Predicted", ha="center", fontsize=20)  # x
            if len(crop_type_list) == 1:
                fig.text(0.35, 0.00, "Predicted", ha="center", fontsize=20)
            elif len(crop_type_list) == 2:
                fig.text(0.5, 0.00, "Predicted", ha="center", fontsize=20)  # x


        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Manually rotate y-axis labels
        for y_label in ax.get_yticklabels():
            y_label.set_rotation(0)


#plt.subplots_adjust(wspace=0.8)  # Reduce vertical space between subplots
plt.subplots_adjust(wspace=0.99)  # Reduce horizontal  space between subplots

# Adjust spacing between subplots
plt.tight_layout(rect=[0.01, 0.1, 0.90, 0.90])
print(OUTFILE )
plt.savefig(OUTFILE.as_posix(), dpi=500)
plt.show()
plt.close()


