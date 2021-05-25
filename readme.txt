==== Preprocessing ====

'tracking_id' is removed entirely for not containing useful information.

'daytime' is converted to 'days_since_new_year' to capture periodicity in the data as a result of seasons.

'blades_angle', 'days_since_new_year' and 'wind_direction' are periodic. For example, 1 degree is very close to 359
degrees even though their difference is large. To capture for this property in the data, for each such feature x, x is
converted to radians if it is in degrees and then replaced with sin(x), cos(x) and the normalised difference between
(sin(x), cos(x)) and (sin(0), cos(0)).

'wind_speed' has two clusters of data points. Only the cluster with a reasonable distribution is kept.

'motor_torque', 'generator_temperature', 'resistance', 'rotor_torque' each consist of two subsets of data that have the
same distribution but are translated and/or rescaled relative to each other. The exact amount is determined through
visual inspection and subsequently corrected. The same pattern is found in 'atmospheric_pressure'. However, correcting
the pattern had a negative impact on model accuracy so the feature is left unchanged.

Negative values in 'blade_length', 'windmill_height' and 'resistance' are replaced with nan.
Values in 'windmill_body_temperature' that are lower than the lowest atmospheric temperature ever recorded are also
replaced with nan.

'blade_area' is created as the product of 'blade_length' and 'blade_breadth'.

'air_density' is created using the air density formula, with a specific gas constant of 287.
Absolute pressure and atmospheric temperature are substituted with 'atmospheric_pressure' and 'atmospheric_temperature'
after converting the latter into kelvins. 'atmospheric_pressure' seemed to represent absolute pressure because its
values are around the absolute atmospheric pressure at sea level.

'air_mass_flow_rate' is created as the product of 'air_density' and 'wind_speed'.

'air_density' and 'air_mass_flow_rate' are created because both are needed to calculate wind turbine power according
to the wind turbine power formula.


Outliers in numerical features are replaced with nan. All features are then imputed using median if numerical else mode.
Then, numerical features are rescaled and categorical features are encoded using one hot encoding.

Feature selection is done by keeping only those with a feature importance higher than the mean feature importance as
determined by lightGBM regressor.


==== Modeling ====

Empty rows and columns, duplicates rows and rows with missing target feature are removed from the dataset.
The model is then trained and scored on the dataset using 10-fold cross validation with the scoring function r2.

Many models were tested. The top three models lightGBM regressor, xgb regressor and random forest regressor were
used to create an ensemble model.

Hyperparameter tuning is then performed on each component model to increase accuracy of the final model.
