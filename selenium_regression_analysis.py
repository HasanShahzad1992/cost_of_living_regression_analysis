import pandas as pd
import selenium.webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib



driver_path="C:/Users/Admin/Desktop/pythonpractice/chromedriver.exe"
driver=wd.Chrome(executable_path=driver_path)
driver.get("https://www.numbeo.com/cost-of-living/rankings_current.jsp")
wait=WebDriverWait(driver,10)
empty_list=[]
for i in range(1,407):
    path="//*[@id='t2']/tbody/tr[" +str(i)+"]/td[2]/a"
    path_cost_living="//*[@id='t2']/tbody/tr[" +str(i)+"]/td[3]"
    path_rent_index="//*[@id='t2']/tbody/tr[" +str(i)+"]/td[4]"
    path_groceries_index="//*[@id='t2']/tbody/tr[" +str(i)+"]/td[6]"
    path_restaurant="//*[@id='t2']/tbody/tr[" +str(i)+"]/td[7]"
    path_local_purchasing="//*[@id='t2']/tbody/tr[" +str(i)+"]/td[8]"
    target_city=wait.until(EC.presence_of_element_located((By.XPATH,path)))
    target_city=target_city.text.split(",")
    target_city=target_city[0]
    target_cost_living_index=wait.until(EC.presence_of_element_located((By.XPATH,path_cost_living)))
    target_rent_index=wait.until(EC.presence_of_element_located((By.XPATH,path_rent_index)))
    target_groceries_index=wait.until(EC.presence_of_element_located((By.XPATH,path_groceries_index)))
    target_restaurant_price=wait.until(EC.presence_of_element_located((By.XPATH,path_restaurant)))
    target_local_purchasing=wait.until(EC.presence_of_element_located((By.XPATH,path_local_purchasing)))
    empty_list.append([target_city,float(target_cost_living_index.text),float(target_rent_index.text),float(target_groceries_index.text),float(target_restaurant_price.text),float(target_local_purchasing.text)])
data_frame=pd.DataFrame(empty_list,columns=["city","living_cost","rent","grocery","restaurant","local_purchase"])
data_frame.to_excel("selenium_regression_country_indexes.xlsx")
data_frame=pd.read_excel("selenium_regression_country_indexes.xlsx")
text_based_features=data_frame["city"]
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
text_based_features_vectorized=vectorizer.fit_transform(text_based_features)
numeric_features=data_frame.drop(["city","living_cost"],axis=1)
combined_text_numeric=hstack([text_based_features_vectorized,numeric_features])
X=combined_text_numeric
#we are picking living_cost as a series, not 2D, but since it is not used for training so no issue
Y=data_frame["living_cost"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
joblib.dump(model,"cost_living.joblib")
print("model has been trained")
loaded_model=joblib.load("cost_living.joblib")
prediction=loaded_model.predict(X_test)
mean_squared_error=mean_squared_error(Y_test,prediction)
r2_score=r2_score(Y_test,prediction)
print(mean_squared_error,r2_score)
dummy_data_frame=pd.DataFrame([["London",60,50,92,79]],columns=["city","rent","grocery","restaurant","local_purchase"])
dummy_text=dummy_data_frame["city"]
vectorizer_dummy_text=vectorizer.transform(dummy_text)
numeric_features=dummy_data_frame.drop(["city"],axis=1)
combined_text_numeric_dummy=hstack([vectorizer_dummy_text,numeric_features])
prediction=loaded_model.predict(combined_text_numeric_dummy)
print(prediction)
import numpy as np
import selenium.webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

driver_path="C:/Users/Admin/Desktop/pythonpractice/chromedriver.exe"
driver=wd.Chrome(executable_path=driver_path)
driver.get("https://www.numbeo.com/property-investment/rankings_current.jsp")
wait=WebDriverWait(driver,10)
city_property_info=[]
target_tr=wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR,"table#t2 tbody tr")))
for i in target_tr:
    target_td=i.find_element(By.CLASS_NAME,"cityOrCountryInIndicesTable")
    target_anchor=target_td.find_element(By.CLASS_NAME,"discreet_link")
    target_city_country=target_anchor.text.split(",")
    target_city=target_city_country[0]
    price_income_ratio=i.find_element(By.CLASS_NAME,"sorting_1")
    target_rest_of_tds=i.find_elements(By.TAG_NAME,"td")
    gross_rental_yield_center=target_rest_of_tds[3]
    gross_rental_yield_outside=target_rest_of_tds[4]
    price_rent_ratio_center=target_rest_of_tds[5]
    price_rent_ratio_outside=target_rest_of_tds[6]
    mortgage_income=target_rest_of_tds[7]
    affordability=target_rest_of_tds[8]
    city_property_info.append([target_city,float(price_income_ratio.text),float(gross_rental_yield_center.text),float(gross_rental_yield_outside.text),float(price_rent_ratio_center.text),float(price_rent_ratio_outside.text),float(mortgage_income.text),float(affordability.text)])

data_frame=pd.DataFrame(city_property_info,columns=["city","price_income_ratio","gross_rental_yield_center","gross_rental_yield_outside","price_rent_ratio_center","price_rent_ratio_outside","mortgage","affordability_index"])
data_frame.to_excel("city_affordability.xlsx")
data_frame=pd.read_excel("city_affordability.xlsx")
text_based_features=data_frame["city"]
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
text_based_features_vectorized=vectorizer.fit_transform(text_based_features)
numerical_features=data_frame.drop(["city","affordability_index"],axis=1)
combined_text_numerical=hstack([text_based_features_vectorized,numerical_features])

X=combined_text_numerical
Y=data_frame["affordability_index"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=DecisionTreeRegressor()
model.fit(X_train,Y_train)
joblib.dump(model,"city_affordability.joblib")
print("model has been trained")
loaded_model=joblib.load("city_affordability.joblib")
prediction=loaded_model.predict(X_test)
r2_score=r2_score(Y_test,prediction)
print(r2_score)
loaded_model=joblib.load("city_affordability.joblib")
dummy_data_frame=pd.DataFrame([["Lahore",74,16,28,19,73,868]],columns=["city","price_income_ratio","gross_rental_yield_center","gross_rental_yield_outside","price_rent_ratio_center","price_rent_ratio_outside","mortgage"])
text_based_features=dummy_data_frame["city"]
text_based_features_vectorized=vectorizer.transform(text_based_features)datasets.cifar10.load_data()
print(text_based_features_vectorized.shape)
numerical_features=dummy_data_frame.drop(["city"],axis=1)
combined_text_numeric=hstack([text_based_features_vectorized,numerical_features])
print(combined_text_numeric.shape)
prediction=loaded_model.predict(combined_text_numeric)
print(prediction)

import tensorflow as tf
from tensorflow.keras import models,layers,datasets
from tensorflow.keras.preprocessing import image
import numpy

(X_train,Y_train),(X_test,Y_test)=datasets.mnist.load_data()
X_train=X_train/255
X_test=X_test/255
model=models.Sequential([layers.Flatten(input_shape=(28,28)),layers.Dense(128,activation="relu"),layers.Dropout(0.2),layers.Dense(10)])
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=5)
model.save("digits_classifier.keras")
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)
print(test_loss,test_accuracy)
loaded_model=models.load_model("digits_classifier.keras")
image_variable=image.load_img("digit_tensor.jpg",target_size=(28,28),color_mode="grayscale")
image_array=image.img_to_array(image_variable)
expand_dimenstions=np.expand_dims(image_array,axis=0)
predictions=loaded_model.predict(expand_dimenstions)
predict_digit=np.argmax(predictions)
print(predict_digit,predictions)

































