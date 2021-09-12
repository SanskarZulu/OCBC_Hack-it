# **Hack-IT OCBC: Final Assessment (Data and AI Track)**
I am [**Sanskar Sharma**](https://www.linkedin.com/in/sanskar-sharma-601aa1195/) B.Tech entrant CSE at MIT Academy of Engineering, Pune. 
This project is my submission as the **Final Assessment of Hack-It organized by OCBC for the Data and AI Track.**

## **Scope:**
* Use the [US Traffic 2015 Dataset](https://www.kaggle.com/jboysen/us-traffic-2015),
publicly available on Kaggle, to
visualise the traffic patterns.
* This assignment aims to clean and
analyse the dataset, create
appropriate models and visualise
them using the proper software.

## **Functionality:**
* Use the appropriate algorithms and models to find out the top 5 most
obvious patterns from this data.
* Support your hypotheses with appropriate data
<br><br>

# **Details of the text files loaded as pandas dataframe are as follows**
1.	**Traffic_df:**<br>
Shape: (7140391,38)<br>
It consisted of traffic count of each hour of each day of year 2015 for various states in US represented with their fips code, station ids, travel lane, functional classes, direction of travel etc.
2.	**Traffic_station_df: (cross-referenced by station ids)**<br>
Shape: (28466,55)<br>
It consisted of deeper insights for each station their geographical coordinates, historical data, lanes & vehicle monitored for traffic, sensors involved, county fips codes, data retrieval methods, vehicle classification algorithms etc. This data was rather more typical to deal with as it contained too many null values, repeated columns and large number of columns.
<br><br>

# **Data Cleaning & Handling**
1.	**Traffic_df:**<br>
* Firstly, I assigned the names for the week days & months in a dictionary for better understanding for the data.
* Dropped the unwanted (like restrictions, had all NAN values), repeated (like direction of travel & direction of travel name) and constant columns (like year is same for all 7.1M rows).
* Converted the 24 hour formatted traffic count columns to 4 columns as:<br>
**NMAE:**<br>
**Night (00-06), Morning (06-12), Afternoon (12-18) and Evening (18-24). [as int type]**<br>
And dropped the 24 hour format traffic count columns.
* The four section binned format (NMAE) contained outliers as 75% of the data was about approx. 1/1000th time of the maximum traffic count. Hence, rows with NMAE values greater than 10k were dropped.<br>
**New shape of traffic_station: (7140391,12)**
2.	**Traffic_station_df:**<br>
* Firstly, dropped the unwanted, repeated and constant columns from dataframe.
* Dropped the columns which contained null values more than 20k as the dataset contains 28k rows. Hence removed columns with 70% above null values.
* Only taken station ids which are present in traffic_df and removed station ids which are not in traffic_df. 
* **Binary Encoded** the following columns as 0 and 1:
hpms sample type, national highway system and sample type for vehicle classification
New shape of traffic_station_df: (24275,15)
<br><br>

# **Patterns/Trends**
1. **45, 9 and 4** are the **heavy traffic states in US** for the year 2015, whereas **56 and 30 are the states where there’s no traffic at all** (Numbers are fips state codes). The **station with 017200, 119780 and 10093 ids have recorded highest mean traffic in US** for the year 2015, whereas **stations ids 00041 and 075040 appears to be have no habit at all**.<br><br>
**Fips state code wise mean traffic count per hour for all four time interval**<br>
<img src="Visualizations\fips_state_code_NMAE_plot.png" alt="fips_state_code_NMAE_plot" style="height: 400px; width:700px;"/>

2.	It is observed that US have **heaviest traffic in the evening (18-24 Hr) and lightest traffic is obtained at night (00-06 Hr)** for the year 2015. Morning mean traffic is second heaviest among four.<br> 
**General trend in mean traffic reduction of a day for year 2015:** <br>
**Evening (18-24) > Morning (06-12) > Afternoon (12-18)> Night (00-06)**
<br>
It can be justified as Office workers, students, daily wage workers etc. returns home in the evening and usually plans to go out then, resulting in heavy evening traffic, morning traffic is accounted by the rush to go to school, offices and for work and usually its bedtime at night resulting in least traffic due to 24x7 transportation services in US.
<img src="Visualizations\Mean_traffic_perH_2015_plot.png" alt="Mean_traffic_perH_2015_plot" style="height: 400px; width:400px;"/>

3.	**Saturdays in US are found to be extreme heavy traffic** day than other days except for night time.<br>
**General trend in traffic for a week is:**<br>
It increases from Monday reaches its extreme count on Saturday and then drops nearer to Monday’s traffic on Sunday. <br>
It can be explained as Saturday and Sunday are the weekends and as Saturday being the first people generally plan this day for some outing, shopping and personal amusements resulting in more traffic while they in general choose to rest at home on Sunday.
<img src="Visualizations\day_of_week_NMAE_plot.png" alt="day_of_week_NMAE_plot" style="height: 400px; width:700px;"/>

4.	There’s a **monthly parabolic trend observed in traffic** for the complete year 2015 in US. **January and February are months with lightest and almost equal traffic**. The traffic increases after February and Reaches its extreme count in month of **July (being heaviest traffic month)** and then starts decreasing gradually till December except for night time (least noticeable parabola)<br>
It can be justified as in US Jan and Feb are the months when its snows blocking almost most of the lanes and routes. Also, people avoids travelling in the snowy season which explains the least traffic in these months. The parabola is due to the seasoning in US. As it is a colder place with its summer season close to June and July and people there prefer this weather to travel and celebrate and what not resulting it to be heaviest traffic month. 
<img src="Visualizations\month_of_data_NMAE_plot.png" alt="month_of_data_NMAE_plot" style="height: 400px; width:700px;"/>

5.	From all the inferences and observations made from various visualizations and analysis. It is found that for various states and station ids in US on all weekdays and months mean traffic on various directions of travel, lanes of travel and functional classifications at the **night time is constant and doesn’t change much**. <br>
In general, across the United States night traffic is almost same, which can be **visualized by the approximately straight lines in the various lines plots** and also seen by the constant colour in heat maps for night time. 
Below are the instances of constant/straight line mean traffic per hour at night with different scenarios. 
<br><br>

# **Data Modelling**

### **Simple Route Prediction Model:**
* **User input:** Weekday of travel, Month of Travel, Fips state code (US)
* **Output:** 5 Most and 5 least traffic scenarios to keep in mind like direction of travel, functional class and lane of travel. 

### **Complex Evening Traffic Prediction Model for UK state with maximum dispersed traffic count:**
**<p style="text-align: center;">Points to Remember:</p>**

> * I have predicted only **evening (18-24 Hr)** traffic as it is maximum compared to others across all states for the year 2015
> * Due to large training set and training time for all the states, this prediction model is built **only for one UK state (i.e with fips 4)**
> * State with **fips 4 has been selected because of its most dispersed traffic data (maximum variance/ standard deviation)** making it difficult for normal humans to predict.
> * Also, the dataset used is a combination of both datasets **traffic_df** and **traffic_station_df** with suitable columns

**1.	Data Preparation:**<br>

(Note: Data cleaning and handling for both traffic_df and traffic_station_df has been done already)
* **Using pivot table and standard deviation** found the US state with most dispersed data i.e. state with “4” as fips id. Filtered both traffic_df and traffic_station_df for fips id “4”.
* **Inner Join both the dataset** using **station_id** as the common key and create a new_df which will be treated as final traffic_df.
* **Drop the repeated and unwanted columns** like station_location, index_x, index_y etc.

**2.	Feature Engineering:**<br>

As all the features were categorical except for longitude, latitude, number of lanes in direction indicated and number of lanes monitored for traffic volume.
Depending on the type of feature, type of categorical features and number of categories in categorical features different encoding techniques were used. 
* **Binary Encoding:**<br> 
**Method of data retrieval** contained two categories as automated and manual which were encoded as 1 and 0 respectively.<br> 
**National highway system** contained two categories as Yes and No which were encoded as 1 and 0 respectively.
* **Label Encoding:** <br>
**Month** is a **qualitative nominal variable** but as the data is of year 2015 only, there are only 12 months of 2015 which can be ordered in a sequence and can be considered as ordinal.<br>
**1-Jan, 2-Feb….12-Dec.**
* **One-Hot Encoding:** <br>
Categories of **day of week, direction of travel, functional classes, lane of travel and type of sensor** were **one-hot encoded** as columns. 
* [**Feature Hash Encoding:**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html)<br>
The **fips state with id “4” contained near about 269 different station ids.** This feature is a nominal categorical feature hence cannot be label encoded and being large in number it can’t be one-hot encoded as well.<br>
The station ids categorical sequences of **symbolic feature names (strings) into SciPy sparse matrices, using a hash function** to compute the matrix column corresponding to a name. Hence, 269 categorical features were converted to **10 station groups columns.**

**3.	Training & Predictions:**

**Sklearn: Random Forest Regressor Class**
* *It uses [**Standard deviation reduction algorithm**](https://www.saedsayad.com/decision_tree_reg.htm#:~:text=The%20standard%20deviation%20reduction%20is,%2C%20the%20most%20homogeneous%20branches)) for **regression based decision tree models**. 
* The object of following Random Forest in-built class in sklearn python library uses **500 decision trees** with **random state as 42** to training the model.
* The **Mean Absolute Error** in evening traffic prediction is **140.24** degrees
* Calculated the **feature importance** for each feature and observed that **out of 40 only 10 features were contributing for the accuracy** of the model. Hence, in future models and trainings other non-contributing features can be dropped as per the usage.


<img src="Visualizations\Feature_Importance_Graph.png" alt="Feature_Importance_Graph" style="height: 500px; width:800px;"/>

## **<p style="text-align: center;">Accuracy of the model: 93.26%</p>**
<br><br>

# **Data Model (A decision tree with depth 3)**
<img src="Data Models\Data_Model_depth_3.png" alt="Feature_Importance_Graph" style="height: 250px; width:1050px;"/>
<br><br>

# **Importance of the above Evening Traffic Prediction Model**
This model can be used for any of the fips state code as desired by the trainer. As the data set contained **7.1M rows which couldn’t be trained completely on my system configurations** hence I chose to build it for only a particular state of US. 

A user have to enter certain details of his travel route, station ids, day & month of travel and the geographical coordinate of that area and in turn the model will return him the traffic conditions in evening for the same with an **accuracy of 93.26%.** So that he can accordingly revise his travel details as per his/her interests. **A better accuracy can be achieved with more data.**
<br><br>

# **More Details**
## **Project Details**
* [Complete Project](https://drive.google.com/drive/folders/1Qb1jJxznypP0l13Z-dNOysE8X1Zw_W-W?usp=sharing)
* [Complete Documentation](https://drive.google.com/file/d/1tDD1OpBtA6FnjGtciYOF3U0LdsWjyCQ-/view?usp=sharing)
* [Top 5 & Additional Patterns](https://drive.google.com/file/d/1k33EbgOfYHp_OQFq-frek1n8OtoCi-Vt/view?usp=sharing)


## **Connect with me**
 
* [LinkedIn](https://www.linkedin.com/in/sanskar-sharma-601aa1195/)















