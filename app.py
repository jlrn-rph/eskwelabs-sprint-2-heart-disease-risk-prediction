import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle

st.set_page_config(page_title='Healing the Heart and the Mind: A Data-Driven Approach to Understanding Heart Disease Health Indicators and Mental Health')

#  Function that loads the data from a CSV file into a Pandas DataFrame
def load_data():
    data = pd.read_csv("./data/2021_brfss_clean_heart_disease_health_indicators.csv")
    return data

# Function that provides an overview of the project, including research objectives, scope, and limitations
# It also displays visualizations of key metrics and displays the dataset
def introduction():
    st.image("./assets/header.png")

    st.subheader("Research Objectives")
    st.image("./assets/objectives.png")

    st.subheader("Methodology")
    st.image("./assets/methodology.png")
    st.markdown(
        """
        **Dataset**

        The Behavioral Risk Factor Surveillance System (BRFSS) is a survey conducted in the U.S. to understand health-related risk behaviors and chronic health conditions of residents in the US.
        * Surveyor: Centers for Disease Control and Prevention
        * Method: Telephone Survey

        **Data Processing**
        """
    )
    st.image("./assets/processing.png")

    data = load_data()
    with st.expander("View Data"):
        st.dataframe(data)
        st.caption("*Source: CDC Behavioral Risk Factor Surveillance System 2021*")

# Function that explores the relationship between age, income, education, and financial inclusion 
# among genders in the Philippines. It provides visualizations of key findings and interprets the results
def data_viz():
    st.title("Exploratory Data Analysis")
    st.info(
        """
        **💡Did you know?**

        **The incidence rate of heart disease for an 80-year old is *50 times (!)* that of an 18-year old!**
        """
    )

    selected = st.selectbox("Which data analysis would you like to see?", ["Demographics", "Health Behaviors", "Risk Factors"])

    if selected == "Demographics":
        tab1, tab2, tab3 = st.tabs(["Age", "Sex", "Income"])

        with tab1:
            st.image("./assets/eda_age.png")
            st.markdown(
                """
                The older you get, the more likely you are to be afflicted by heart disease. Here's why: 

                As people age, the heart and blood vessels undergo various changes such as **thickening and stiffening of the arteries**, **reduced elasticity of blood vessels**, and **decreased pumping ability of the heart**, contributing to the likelihood of heart disease. 

                *Source: American Heart Association. (2021). Aging and heart disease. Retrieved from https://www.heart.org/en/health-topics/heart-attack/understand-your-risks-to-prevent-a-heart-attack/aging-and-heart-disease.*
                """
            )

        with tab2:
            st.image("./assets/eda_sex.png")
            st.markdown(
                """
                Our dataset shows that **males are about 50% more likely than females** to be afflicted by heart disease. This is likely due to:

                1. **Lifestyle factors**. Men are more likely to engage in certain lifestyle behaviors that increase the risk of heart disease, such as **smoking and excessive alcohol consumption**.
                2. **Hormonal differences**. Estrogen, a female sex hormone, helps to keep blood vessels flexible and can reduce the buildup of plaque in the arteries. **As women go through menopause, their estrogen levels decrease, which can increase the risk of heart disease**. 

                *Source: Mendelsohn, M. E., & Karas, R. H. (2005). The protective effects of estrogen on the cardiovascular system. New England Journal of Medicine, 340(23), 1801-1811.*
                """
            )

        with tab3:
            st.image("./assets/eda_income.png")
            st.markdown(
                """
                Generally, the richer you are, the less likely you are to be afflicted by heart disease. Here's why: 

                1. **Access to healthcare**. Wealthier individuals are more likely to have access to quality healthcare, improving their chances of receiving timely treatment for heart disease risk factors.
                2. **Less occupational exposure**. Wealthier individuals are less likely to be occupationally exposed to hazardous pollutants or chemicals, which can increase the risk of heart disease.

                *Source: Mackenbach, J. P., Stirbu, I., Roskam, A. J., Schaap, M. M., Menvielle, G., Leinsalu, M., ... & Kunst, A. E. (2008). Socioeconomic inequalities in health in 22 European countries. New England Journal of Medicine, 358(23), 2468-2481.*
                """
            )

    elif selected == "Health Behaviors":
        tab1, tab2, tab3 = st.tabs(["Difficulty Walking", "Physical Activity", "Alcohol Consumption"])

        with tab1:
            st.image("./assets/eda_diffwalk.png")
            st.markdown(
                """
                Interestingly, those with difficulty walking are **thrice as likely** to be afflicted by heart disease than those who don't. Key possible reasons are:

                1. **Higher rates of obesity**. Difficulty walking contributes to obesity, which in turn increases the risk of **high blood pressure**, and other cardiovascular risk factors.
                2. **Higher rates of diabetes**. Those with difficulty walking are more likely diabetic, which itself is a major risk factor for heart disease, as it can **damage blood vessels and increase the risk of atherosclerosis**.

                *Source: Guralnik, J. M., Ferrucci, L., Simonsick, E. M., Salive, M. E., & Wallace, R. B. (1995). Lower-extremity function in persons over the age of 70 years as a predictor of subsequent disability. New England Journal of Medicine, 332(9), 556-561.*
                """
            )

        with tab2:
            st.image("./assets/eda_physhlth.png")
            st.markdown(
                """
                People who engage in physical activity are almost **less than twice as likely** to have heart disease compared to people who don't. 

                This is unsurprising as physical activity contributes to **improved cardiovascular health**, strengthening the heart and blood vessels, improving blood flow, and reducing inflammation.

                *Source: American Heart Association. (2021). Physical Activity Improves Quality of Life. Retrieved from https://www.heart.org/en/healthy-living/fitness/fitness-basics/aha-recs-for-physical-activity-in-adults.*
                """
            )

        with tab3:
            st.image("./assets/eda_hvyalc.png")
            st.markdown(
                """
                This is a particularly interesting result, where those who **consume alcohol are almost less than twice as likely to contract heart disease than teetotalers**.

                There is some evidence to suggest that **moderate alcohol consumption can lower the risk of heart disease**. 

                One possible theory is that alcohol consumption improves insulin sensitivity, which can help to **lower the risk of type 2 diabetes**. 

                Diabetes is a known risk factor for heart disease. 

                *Source: Gaziano, J. M. (2021). Moderate alcohol intake and cardiovascular disease. Harvard Health Publishing. Retrieved from https://www.health.harvard.edu/blog/moderate-alcohol-intake-and-cardiovascular-disease-2021030522361.*
                """
            )

    elif selected == "Risk Factors":
        tabs1, tabs2, tabs3, tabs4, tabs5, tabs6 = st.tabs(["Cholesterol", "Diabetes", "High Blood Pressure", "Stroke", "Depressive Disorder", "Obesity"])

        with tabs1:
            st.image("./assets/eda_highchol.png")
            st.markdown(
                """
                While our dataset shows that those without high cholesterol are more likely to be afflicted by heart disease, this is contrary to the scientific evidence on the same subject.

                This is likely due to the fact that **cases of high cholesterol are often underreported**. 

                A JAMA Cardiology study in 2013 found that **a significant proportion of established heart disease patients with high cholesterol were not aware of their condition** and were not receiving treatment, highlighting the need for improved efforts to identify high cholesterol. 

                *Source: Coutinho, T., Goel, K., Correa de Sa, D., Carter, R. E., Hodge, D. O., Kragelund, C., ... & Lopez-Jimenez, F. (2013). Combining body mass index with measures of central obesity in the assessment of mortality in subjects with coronary disease: role of "normal weight central obesity". JAMA cardiology, 148(7), 717-727.*
                """
            )

        with tabs2:
            st.image("./assets/eda_diabetes.png")
            st.markdown(
                """
                People with diabetes are **more than twice as likely to have heart disease compared to people without diabetes**. This could be because diabetes contributes to 

                1. **High blood glucose levels**, which can damage the blood vessels and lead to atherosclerosis (hardening and narrowing of the arteries)
                2. **High blood pressure**, which is a major risk factor for heart disease.


                *Source: American Diabetes Association. (2021). Diabetes and Cardiovascular Disease. Retrieved from https://www.diabetes.org/diabetes/complications/cardiovascular-disease.*
                """
            )

        with tabs3:
            st.image("./assets/eda_highbp.png")
            st.markdown(
                """
                People with high blood pressure are **more than four times as likely to have heart disease compared to people without high blood pressure**. This could be due to 

                1. **Increased workload on the heart**, forcing the heart to work harder to pump blood leading to an enlarged heart, weakening of the heart muscle, and eventually heart failure.
                2. **Increased risk of blood clots**, which can lead to heart attack or stroke, which itself is a likely indicator of heart disease.

                *Source: Mayo Clinic Staff. (2021). High blood pressure (hypertension). Mayo Clinic. Retrieved from https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410.*
                """
            )

        with tabs4:
            st.image("./assets/eda_stroke.png")
            st.markdown(
                """
                People with stroke are almost **more than four times as likely to have heart disease** compared to people without stroke. This could be due to 

                1. **Shared risk factors**. As we have seen so far, many of the risk factors for stroke and heart disease are the same, such as **high blood pressure**, **diabetes**, and **smoking**. Having one of these risk factors increases the risk of developing both stroke and heart disease.
                2. Stroke can **damage the blood vessels** in the brain and body, increasing the risk of developing **atherosclerosis** (buildup of fatty deposits in arteries) and hence heart disease.

                *Source: American Heart Association. (2021). Stroke and heart disease. Retrieved from https://www.heart.org/en/health-topics/consumer-healthcare/what-is-cardiovascular-disease/stroke-and-heart-disease.*
                """
            )

        with tabs5:
            st.image("./assets/eda_depress.png")
            st.markdown(
                """
                People experiencing **depression**, anxiety, stress, and even PTSD over a long period of time may experience certain physiologic effects on the body, such as increased cardiac reactivity reduced blood flow to the heart, and heightened levels of cortisol.

                *Source: Center for Disease Control and Prevention (2020). Heart Disease and Mental Health Disorders. Retrieved from Heart Disease and Mental Health Disorders | cdc.gov.*
                """
            )

        with tabs6:
            st.image("./assets/eda_obese.png")
            st.markdown(
                """
                **1 in 10 people with CLASS 2 OBESITY experienced heart disease**

                The extra weight in obese people forces the heart to do more work. It can also cause problems by increasing the risk of developing many other factors that make heart disease more likely.

                *Source: National Institute of Diabetes and Digestive and Kidney Diseases. (2021). Heart disease and stroke. Retrieved from https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke*
                """
            )


def model_interpret():
    st.title("Model Selection and Interpretability")        

    st.header("Model Selection")
    st.image("./assets/model.png")
    st.markdown(
        """
        Three algorithms were considered in creating the predictive model, these are **Logistic Regression**, **K Nearest Neighbors**, and **Gradient Boosting**. To address the data imbalance, an under-sampling is performed using **Random Under Sampler**.

        The table above shows recall and accuracy scores across train, validation, and holdout sets.

        Amongst the three models, the one with the best recall scores was selected. This is because our target variable to predict is having heart disease and what we want is to create a model that would correctly identify those with heart disease.
        """
    )

    st.header("Model Interpretability")
    st.subheader("SHAP Waterfall Plot")
    st.image("./assets/shap_waterfall.png")
    st.markdown(
        """
        The SHAP waterfall plot shows the contribution of each feature toward the final prediction of the model.

        In this case, the **high blood pressure** feature has a positive contribution of 0.4, meaning that having high blood pressure increases the prediction score. Similarly, those having **difficulty walking** and those belonging to the **age group of 65-69** also have a positive contribution of 0.36 and 0.32 respectively.

        On the other hand, being female (sex) and being inactive in physical activities have negative contributions of -0.34 and- 0.11 respectively. This means that being female and being inactive decrease the prediction score.

        Other features that have notable contributions to the prediction score are very good general health and depressive disorder.
        """
    )

    st.subheader("SHAP Beeswarm Plot")
    st.image("./assets/shap_beeswarm.png")
    st.markdown(
        """
        This SHAP beeswarm plot shows the impact of different feature values on the model output. The vertical axis represents the features, and the horizontal axis represents the SHAP value, which is a measure of the impact of that feature on the model output.

        Based on the SHAP beeswarm plot, the features that have a high positive impact on the model output are **sex**, **high blood pressure**, and **age groups 80+**, and **75-79**. This means that having these feature values increases the model prediction score, meaning that **increases the risk of experiencing heart disease**.

        On the other hand, the features that have a negative impact on the model output are excellent general health, age groups 30-34, and 40-44. This means that having these feature values decreases the model prediction score, meaning that it decreases the risk of experiencing heart disease.

        Also, being diagnosed with a **depressive disorder** contributes to a **higher risk of contracting heart disease**.

        Overall, this plot helps to identify which feature values have the largest impact on the model output, and it can be used to gain insight into the behavior of the model.
        """
    )

def predictive_modeling():
    # Load the trained model
    model = pickle.load(open('./model/model.pkl', 'rb'))

    # Load the column names used for training
    columns = pd.read_csv('./model/training_columns.csv')['columns'].tolist()

    def one_hot_encode_input(data, columns):
        # One-hot encode the categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        encoded_data = pd.get_dummies(data, columns=categorical_cols)
        
        # Add any missing columns
        missing_cols = set(columns) - set(encoded_data.columns)
        for col in missing_cols:
            encoded_data[col] = 0
        
        # Reorder columns to match training data
        encoded_data = encoded_data[columns]
        
        return encoded_data


    # Define the Streamlit app
    st.title('Heart Disease Risk Prediction')
    st.markdown(
        """
        The heart disease risk prediction model is a machine learning powered tool using Gradient Boosting algorithm that uses various patient characteristics to predict the risk of developing heart disease. In order to predict your risk of heart disease, provide your information in terms of demographics, health behavior, risk factor, mental health status, and healthcare access. 
        """
    )
    st.warning(":warning: Note that this is not a replacement for a diagnosis and you should consult your doctor if you experience any symptoms.")

    st.subheader("Demographics")
    sex = st.radio("What is your biological sex?", ["Male", "Female"])
    age = st.selectbox("Which age group do you belong?", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                     "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 and up"])
    education = st.selectbox("What is your highest educational attainment?", 
                             [" Never attended school / Only Kindergarten", "Grades 1-8 (Elementary)", "Grades 9-11 (Some High School)", "Grade 12 or GED (High School Graduate)", "College 1-3 (Some College/Technical School)", "College Graduate"])
    income = st.selectbox("Which is your income bracket?", ["Less than 10k USD", "10k-15k USD", "15k-20k USD", "20k-25k USD", "25k-35k USD", "35k-50k USD", "50k-75k USD", "75k-100k USD", "100k-150k USD", "150k-200k USD", "200k USD or more"])

    st.subheader("Health Behaviors")
    general_health = st.selectbox("How will you describe your general health?", ["Poor", "Fair", "Good","Very Good", "Excellent"])
    phys_hlth = st.selectbox("Do you have a poor physical health?", ["Yes", "No"])
    diff_walk = st.selectbox("Do you have difficulty in walking?", ["Yes", "No"])
    phys_activity = st.selectbox("Are you physically active?", ["Yes", "No"])
    fruits = st.selectbox("Do you eat 1 fruit per day?", ["Yes", "No"])
    veggies = st.selectbox("Do you eat 1 vegetable per day?", ["Yes", "No"])
    hvy_alcohol = st.selectbox("Do you have heavy alcohol consumption?", ["Yes", "No"])

    st.subheader("Risk Factors")
    high_bp = st.selectbox("Do you have a history of high blood pressure?", ["Yes", "No"])
    high_chol = st.selectbox("Do you have a history of high cholesterol?", ["Yes", "No"])
    chol_check = st.selectbox("Have you checked your cholesterol levels in the 5 last years?", ["Yes", "No"])
    smoker = st.selectbox("Do you smoke?", ["Yes", "No"])
    stroke = st.selectbox("Do you have a history of stroke?", ["Yes", "No"])
    bmi = st.selectbox("What is your BMI?", [
                            "Underweight", "Normal", "Overweight", "Obese 1", "Obese 2", "Obese 3"])
    diabetes = st.selectbox("What is your diabetes status?", [
                            "No or During Pregnancy", "Pre-Diabetes/Borderline", "Existing Diabetes"])

    st.subheader("Mental Health Status")
    ment_hlth = st.selectbox("How would you describe your mental health status?", 
                             ["0 days not good ", "1-13 days not good ", "14+ days not good"])
    depressive_disorder = st.selectbox("Depressive disorder?", ["Yes", "No"])

    st.subheader("Healthcare Access")
    health_insurance = st.selectbox("Do you have health insurance?", ["Yes", "No"])
    no_doc_cost = st.selectbox("Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it?", ["Yes", "No"])

    # Convert categorical features to binary indicators
    sex = 1 if sex == "Male" else 0
    phys_hlth = 1 if phys_hlth == "Yes" else 0
    diff_walk = 1 if diff_walk == "Yes" else 0
    phys_activity = 1 if phys_activity == "Yes" else 0
    fruits = 1 if fruits == "Yes" else 0
    veggies = 1 if veggies == "Yes" else 0
    hvy_alcohol = 1 if hvy_alcohol == "Yes" else 0
    high_bp = 1 if high_bp == "Yes" else 0
    high_chol = 1 if high_chol == "Yes" else 0
    chol_check = 1 if chol_check == "Yes" else 0
    smoker = 1 if smoker == "Yes" else 0
    stroke = 1 if stroke == "Yes" else 0
    ment_hlth = 1 if ment_hlth == "Yes" else 0
    depressive_disorder = 1 if depressive_disorder == "Yes" else 0
    health_insurance = 1 if health_insurance == "Yes" else 0
    no_doc_cost = 1 if no_doc_cost == "Yes" else 0


    def predict_heart_disease():
        # Create a dictionary to store the user input
        user_input = {
        'Sex': sex,
        'Age': age,
        'Education': education,
        'Income': income,
        'General Health': general_health,
        'Physical Health Problems': phys_hlth,
        'Difficulty Walking': diff_walk,
        'Physical Activity': phys_activity,
        'Fruits': fruits,
        'Vegetables': veggies,
        'Heavy Alcohol Consumption': hvy_alcohol,
        'High Blood Pressure': high_bp,
        'High Cholesterol': high_chol,
        'Cholesterol Check': chol_check,
        'Smoker': smoker,
        'Stroke': stroke,
        'BMI Category': bmi,
        'Diabetes Category': diabetes,
        'Mental Health Issues': ment_hlth,
        'Depressive Disorder': depressive_disorder,
        'Health Insurance': health_insurance,
        'Doctor Visit Affordability': no_doc_cost,
        }

        # Convert the user input into a Pandas DataFrame
        user_input_df = pd.DataFrame(user_input, index=[0])

        # Check for missing values
        if user_input_df.isnull().values.any():
            st.write('Please fill in all the input fields.')
        else:
            # One-hot encode the categorical features in the user input
            encoded_user_input = one_hot_encode_input(user_input_df, columns)

            #Impute any missing values
            imputer = SimpleImputer(strategy='median')
            user_input_imputed = imputer.fit_transform(encoded_user_input)

            # Make a prediction using the trained model
            prediction = model.predict(user_input_imputed)

            # Display the prediction
            if prediction[0]==0:
                st.success(':heart: You are **not at risk** for heart disease!')
                st.markdown(
                    """
                    **What can you do to maintain your healthy heart?**
                    1. **Don't smoke or use tobacco**
                    2. **Get moving**. Aim for at least 30 to 60 minutes of activity daily
                    3. **Eat a heart-healthy diet**
                    4. **Maintain a healthy weight**.  Excess weight can lead to conditions that increase the chances of developing heart disease — including high blood pressure, high cholesterol and type 2 diabetes.
                    5. **Get good quality sleep**. People who don't get enough sleep have a higher risk of obesity, high blood pressure, heart attack, diabetes and depression.
                    6. **Manage stress**. Finding alternative ways to manage stress — such as physical activity, relaxation exercises or meditation — can help improve your health.
                    7. **Get regular health screenings**. High blood pressure and high cholesterol can damage the heart and blood vessels. But without testing for them, you probably won't know whether you have these conditions.
                    
                    *Source: Mayo Clinic. (2022). Strategies to prevent heart disease. Retrieved from https://www.nhs.uk/conditions/coronary-heart-disease/prevention/*
                    """
                )
            else:
                st.error('💔 You are **at risk** for heart disease!')
                st.markdown(
                    """
                    **What can you do to reduce the risk?**
                    1. **Eat a healthy, balanced diet**. A low-fat, high-fibre diet is recommended, which should include plenty of fresh fruit and vegetables (5 portions a day) and whole grains.
                    2. **Be more physically active**. Regular exercise will make your heart and blood circulatory system more efficient, lower your cholesterol level, and also keep your blood pressure at a healthy level.
                    3. **Keep to a healthy weight**. A GP or practice nurse can tell you what your ideal weight is in relation to your height and build. 
                    4. **Give up smoking**. If you smoke, giving up will reduce your risk of developing a coronary heart disease.
                    5. **Reduce your alcohol consumption**
                    6. **Keep your blood pressure under control**. You can keep your blood pressure under control by eating a healthy diet low in saturated fat, exercising regularly and, if needed, taking medicine to lower your blood pressure.
                    7. **Keep your diabetes under control**. Being physically active and controlling your weight and blood pressure will help manage your blood sugar level.
                    8. **Take any prescribed medicine**. It's vital you take it and follow the correct dosage. Do not stop taking your medicine without consulting a doctor first, as doing so is likely to make your symptoms worse and put your health at risk.
                    
                    *Source: National Health Service. (2020). Prevention-Coronary heart disease. Retrieved from https://www.nhs.uk/conditions/coronary-heart-disease/prevention/*
                    """
                )

    st.subheader("Ready to know if you are at risk of heart disease?")
    if st.button('Predict'):
        predict_heart_disease()


# Function to define the conlusion and recommendation
def conclusion():
    st.title("Conclusion and Recommendation")

    # Conclusion
    st.subheader("Heart Disease and Mental Health")
    st.markdown(
        """        
        The study showed that men are more likely to experience heart disease as opposed to women. The factors that could increase the risk of heart disease are **age**, **high blood pressure**, **history of stroke**, **smoking**, **history of diabetes**, and **difficulty walking**.

        Also, we saw that having been diagnosed with depressive disorder in fact could lead to experiencing heart disease as well.

        From this, we see that it is important that each individual to take care of their health as most diseases are related to one another.

        Moreover, as much as we take care of physical health and well-being, people should also be mindful of their mental health as this could lead to complications and manifest as a physical health disease.
        """
    )

    # Recommendation 1
    st.subheader("Recommendation for Policy Makers")
    st.markdown(
        """
        Policy makers across the globe can leverage on this study by creating certain policies that would address the threat of heart disease on the general population.

        In the Philippines, the leading cause of death in 2021 is ischemic heart disease. In fact, this has been the highest cause of death since 2019. 

        Policy makers and the department of health could focus on
        * Information campaigns on the causes/factors of heart disease
        * Availability of medicine for hypertension and diabetes
        * Accessibility to mental health care i.e., free consultations and penetration in rural areas
        """
    )

    # Recommendation 1
    st.subheader("Recommendation for Medical Institutions")
    st.markdown(
        """
        Medical institutions across the globe can leverage on this study creating programs that would improve patient care.

        These are things institutions can consider are:
        * **Develop a personalized risk assessment tool**. While there are many risk factors for heart disease, each individual's risk profile is unique. Health institutions could develop a personalized risk assessment tool that takes into account a wide range of factors, including genetic predisposition, lifestyle choices, and access to healthcare. This could help healthcare providers develop more targeted and effective treatment plans.
        * **Develop targeted cholesterol screening and documentation programs**, especially for high-risk populations such as those with a family history of heart disease or other risk factors. This can help identify and manage high cholesterol levels, which is a major risk factor for heart disease.
        * **Incorporate mental health services during checkups**. Incorporate mental health services into heart disease prevention programs, as depression and anxiety have been linked to an increased risk of heart disease. This can include counseling services, stress reduction techniques, and mindfulness-based interventions.
        * **Implement a telemedicine program specifically for heart disease patients**. This could include regular check-ins with healthcare providers, remote monitoring of vital signs, and virtual support groups. Telemedicine could be particularly helpful for low-income individuals who may have difficulty accessing traditional healthcare services.
        """
    )

    st.image('./assets/message.png')

# Meet the team
def the_team():
    st.title("The Team")
    st.markdown(
        """
        We are the team **lockJAW**! We are a group of individuals from diverse backgrounds who came together as part of the Eskwelabs Data Science Cohort 11. In our second sprint, we collaborated to create a data-driven presentation on the relationship of heart disease and mental health entitled **Healing the Heart and Mind: A Data-Driven Approach to Understanding Heart Disease Health Indicators and Mental Health**. 
        
        We focused on investigating the relationship between mental health and heart disease, as well as exploring the influence of various demographic variables, risk factors, health behaviors, and healthcare access on both mental health and heart disease. In addition, the study aimed to develop predictive models to determine the risk of heart disease based on mental health status and other risk factors. The ultimate goal was to provide recommendations for policymakers and healthcare providers on how to improve mental health and reduce the risk of heart disease. This study represents a critical step towards achieving SDG 3 and a healthier, happier future for all.

        The project uses data from the CDC's 2021 Behavioral Risk Factor Surveillance System data, which is wrangled and analyzed using Python Pandas, exploratory data analysis using Matplotlib, and machine learning algorithm using Gradient Boosting.
        """
    )
    st.header("Members")
    st.subheader("[Austin Loi Carvajal](https://www.linkedin.com/in/austincarvajal)")
    st.markdown(
        """
        * Led the charge in the overall analysis and presentation of results, ensuring that our findings were clearly and persuasively communicated to our audience
        * Utilized machine learning modeling techniques including *Logistic Regression* to develop a predictive model for heart disease risk
        * Developed recommendations and policy implications for policymakers and healthcare institutions to guide policymakers in creating effective policies and programs, and healthcare institutions in developing targeted interventions for individuals at risk
        """
    )

    st.subheader("[Justin Louise Neypes](https://www.linkedin.com/in/jlrnrph/)")
    st.markdown(
        """
        * Spearheaded the design and deployment of the Sprint 2 project on Streamlit, showcasing our findings to others
        * Utilized machine learning modeling techniques including *Gradient Boosting* and *K Neighbors* to develop a predictive model for heart disease risk
        * Employed model interpretability tools like *SHAP beeswarm and waterfall plots* to gain insights into the complex relationships between features and target variable as a reference to the actionable recommendations for healthcare providers and policymakers
        """
    )

    st.subheader("Walter Boo")
    st.markdown(
        """
        * Conducted comprehensive research and exploratory data analysis on different heart disease-related risk factors to ensure that our findings are reliable and actionable for healthcare professionals, policymakers, and individuals looking to reduce their risk of heart disease
        """
    )

# Define the main menu
list_of_pages = [
    "Introduction",
    "Exploratory Data Analysis",
    "Model Selection & Interpretability",
    "Predictive Modeling",
    "Conclusion",
    "The Team"
]

st.sidebar.title(':scroll: Main Menu')
selection = st.sidebar.radio("Go to: ", list_of_pages)

if selection == "Introduction":
    introduction()

elif selection == "Exploratory Data Analysis":
    data_viz()

elif selection == "Model Selection & Interpretability":
    model_interpret()

elif selection == "Predictive Modeling":
    predictive_modeling()

elif selection == "Conclusion":
    conclusion()

elif selection == "The Team":
    the_team()
