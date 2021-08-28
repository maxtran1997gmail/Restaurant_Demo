#!/usr/bin/env python
# coding: utf-8

# # Capstone Project - Opening a Vietnamese Restaurant in HCMC
# ### Data Science Capstone 

# ## Table of contents
# * [Introduction: Business Problem](#introduction)
# * [Data](#data)
# * [Methodology](#methodology)
# * [Analysis](#analysis)
# * [Results and Discussion](#results)
# * [Conclusion](#conclusion)

# 
# ## Introduction: Business Problem <a name="introduction"></a>

# Trong project này em sẽ tìm một địa điểm thích hợp để mở một quán ăn Việt Nam bằng Data Science. Đồ án này hướng đến những người có ý định mở một nhà hàng Việt Nam tại TPHCM nhưng không biết nên mở ở quận nào. 
# 
# Vì hiện tại số lượng nhà hàng Việt Nam ở TPHCM là nhiều vô kể , vậy nên trong Đồ Án này em sẽ tìm ra những khu vực mà có mật độ cạnh tranh quán ăn Việt Nam thấp nhất để giảm thiểu rủi ro cạnh tranh, ngoài ra em còn xem xét đến những yếu tố khách quan như chi phí bất động sản ( nếu thuê, hoặc mua địa điểm kinh doanh), mật độ dân số ( chẳng ai đi mở quán ăn kinh doanh ở nơi không có người ) và mật độ công ty / nhà xưởng /... ( để bán cho công nhân đến ăn sáng / ăn trưa/ ăn tối hoặc ghé quán để " lai rai".
# 
# Những findings và clusters tìm được trong bài làm sẽ được giải thích cặn kẽ cùng với những mặt tốt và hạn chế. 

# ## Data <a name="data"></a>

# Những data cần thu thập là : 
#     
#     - Số quận huyện của TPHCM, cùng với kinh độ vĩ độ ( để vẽ lên bản đồ)
#     - Mật Độ dân số , số công ty , nhà máy , xí nghiệp của từng quận.
#     - Mật độ những địa điểm của từng quận huyện của TPHCM ( sẽ sử dụng Google Places API và Foursquare API )

#  
# # **Cài Đặt Những Thư Viện Cần Thiết ** 
# 
# 
# 
# 

# In[101]:


# In[102]:



# In[104]:

import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from bs4 import BeautifulSoup
import requests
import folium 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from unidecode import unidecode
import geocoder


# # CRAWL DATA

# **Đầu tiên là lấy danh sách những quận huyện của TPHCM **

# In[105]:


data = requests.get("https://en.wikipedia.org/wiki/Category:Districts_of_Ho_Chi_Minh_City").text
soup = BeautifulSoup(data, 'html.parser')
DistrictList = []
for row in soup.find_all("div", class_="mw-category")[0].findAll("li"):
    DistrictList.append(row.text.split(",")[0])
    #DistrictList.append(row.text.rsrip(",")[0])


# In[106]:


DistrictList.pop(0)
DistrictList.append("District 2")
DistrictList.append("District 9")



# In[107]:


import re
patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}

def convert(text):
    """
    Convert from 'Tieng Viet co dau' thanh 'Tieng Viet khong dau'
    text: input string to be converted
    Return: string converted
    """
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
    return output



# In[108]:


District_List= []
for row in DistrictList:
  row = convert(row)
  District_List.append(row)


# In[109]:
# In[110]:


#natural sort order
from natsort import natsorted
District_List_Final = natsorted(District_List)


# In[111]:


District_List_Final[-1] = "Thu Duc District"


# In[112]:


District_List_Final[-2] = "Tan Phu District"


# In[113]:


#District_List_Final


# # Lấy tọa độ ( kinh độ , vĩ độ ) của từng quận 

# In[114]:


hcmc_df = pd.DataFrame(District_List_Final,columns =['District'])


# In[115]:


# define a function to get coordinates
def get_latlng(neighborhood):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Ho Chi Minh, VietNam'.format(neighborhood))
        lat_lng_coords = g.latlng
    return lat_lng_coords


# In[116]:


coords = [ get_latlng(neighborhood) for neighborhood in hcmc_df["District"].tolist() ]



# In[117]:


# create temporary dataframe to populate the coordinates into Latitude and Longitude
df_coords = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])
hcmc_df['Latitude'] = df_coords['Latitude']
hcmc_df['Longitude'] = df_coords['Longitude']


# In[118]:





# In[119]:


# Chỉnh sửa tọa độ một vài điểm trung tâm quận lại cho hợp với logic, do có những quận mà trung tâm tọa độ của quận lại là ở sát 1 quận khác. 
#Tan Phu
hcmc_df.at[22,'Latitude'] = 10.790069
hcmc_df.at[22,'Longitude'] = 106.628524
#Tan Binh
hcmc_df.at[21,'Latitude'] = 10.801484
hcmc_df.at[21,'Longitude'] = 106.654077
#District 7
hcmc_df.at[11,'Latitude'] = 10.738291
hcmc_df.at[11,'Longitude'] = 106.718292
#Thu Duc
hcmc_df.at[23,'Latitude'] = 10.8509551
hcmc_df.at[23,'Longitude'] = 106.7539414


# In[120]:





# # Lấy mật độ dân số của từng Quận

# In[121]:


pd.read_html('https://rentapartment.vn/dan-so-dien-tich-quan-tphcm/')


# In[122]:


population = pd.read_html('https://rentapartment.vn/dan-so-dien-tich-quan-tphcm/')


# In[123]:


#print(f'Total tables: {len(population)}')


# In[124]:


population_a = population[0]



# In[125]:


population_a.rename(columns={"Quận": "District", "Dân số (người)": "Population","Diện tích (km²)" : "Square","Số Phường/Xã":"Ward","Mật độ dân số (người/km²)":"Density"},inplace=True)


# In[126]:





# In[127]:


population_a.rename(columns = {population_a.columns[4]:"Density"},inplace=True)


# In[128]:





# In[129]:


population_b = population[1]



# In[130]:


population_b['Số Phường/Xã'] = 0 


# In[131]:





# In[132]:


population_b.rename(columns={"Huyện": "District", "Dân số (người)": "Population","Diện tích (km²)" : "Square","Số Phường/Xã":"Ward","Mật độ dân số (người / km²)":"Density"},inplace=True)


# In[133]:





# In[134]:


population_b.rename(columns = {population_b.columns[3]:"Density"},inplace=True)


# In[135]:


population_b = population_b[['District','Population','Square','Ward','Density']]


# In[136]:





# In[137]:





# In[138]:


population_final = pd.concat([population_a, population_b],ignore_index=True)



# In[139]:


population_final.replace('Quận','District',regex=True,inplace=True)


# In[140]:


population_final.replace('Huyện','',regex=True,inplace=True)


# In[141]:





# In[142]:





# In[143]:


#Chỉnh lại tên cho đồng nhất
i = 0 
for i in range(len(population_final['District'])):
  data = convert(population_final['District'][i])
  population_final.at[i,'District'] = data
  i+=1 


# In[144]:





# In[145]:


#population_final['District'][12:24]


# In[146]:


i = 12 
while i < 24:
  population_final.at[i,'District'] = population_final['District'][i] + ' District'
  i+=1


# In[147]:





# In[148]:




# In[149]:


#Remove white spaces before text
i = 0 
for i in range(len(population_final['District'])):
  population_final.at[i,'District'] = population_final['District'][i].strip()
  i+=1


# In[150]:





# In[151]:


hcmc_df = hcmc_df.merge(population_final,on='District',how='inner')


# In[152]:





# # Lấy dữ liệu giá nhà

# In[153]:


source_housing_price = requests.get("https://mogi.vn/gia-nha-dat").text
soup = BeautifulSoup(source_housing_price, 'lxml')
table_housing_price = soup.find("div", class_="mt-table")


# In[154]:


table_rows = table_housing_price.find_all("div", class_="mt-row")
res_housing_price = []
for tr in table_rows:
    district = tr.find("div", class_="mt-street").a.text
    medium_price = tr.find("div", class_="mt-vol").span.text
    row = [district, medium_price]
    res_housing_price.append(row)


# In[155]:


df_housing_price = pd.DataFrame(res_housing_price, 
                                columns=["District", "Average Housing Price (1M VND)/m2"])


# In[156]:


df_housing_price = df_housing_price.reset_index().drop("index", axis=1)

# Remove the word "Quận"
df_housing_price["District"] = ( df_housing_price["District"]
                                        .str.replace("\n", "").str.replace("Quận", "").str.replace("Huyện", "")
                                        .str.strip() 
                                   )

# Remove Vietnamese accents
df_housing_price["District"] = df_housing_price["District"].apply(unidecode)

# Remove the word "triệu" (It's 10^6 in Vietnamese)
df_housing_price["Average Housing Price (1M VND)/m2"] = ( df_housing_price["Average Housing Price (1M VND)/m2"]
                                                .str.replace("triệu", "")
                                                 .str.replace(",", ".").str.replace(" /m2", "")
                                                 .str.strip()
                                            )



# In[157]:


df_housing_price['District'][1] = df_housing_price['District'][1].replace("(TP. Thu Duc)","")
df_housing_price['District'][8] = df_housing_price['District'][8].replace("(TP. Thu Duc)","")
df_housing_price['District'][18] = df_housing_price['District'][18].replace("TP. Thu Duc","Thu Duc")


# In[158]:





# In[159]:


i = 0
while i < 12:
  df_housing_price.at[i,'District'] = 'District ' + df_housing_price['District'][i] 
  i+=1


# In[160]:


i = 12
while i < 24:
  df_housing_price.at[i,'District'] =  df_housing_price['District'][i] + ' District'
  i+=1


# In[161]:


#Remove white spaces before text
i = 0 
for i in range(len(df_housing_price['District'])):
  df_housing_price.at[i,'District'] = df_housing_price['District'][i].strip()
  i+=1


# In[162]:


#Remove white spaces before text
i = 0 
for i in range(len(df_housing_price['Average Housing Price (1M VND)/m2'])):
  df_housing_price.at[i,'Average Housing Price (1M VND)/m2'] = df_housing_price['Average Housing Price (1M VND)/m2'][i].strip()
  i+=1


# In[163]:





# In[164]:


hcmcdf_with_houseprice= hcmc_df.merge(df_housing_price,on='District',how='inner')


# In[165]:





# # Lấy dữ liệu số lượng công ty / doanh nghiệp từng Quận

# In[166]:


url = 'https://thongtindoanhnghiep.co/tim-kiem?location=%2Ftp-ho-chi-minh&kwd='


# In[167]:


def get_page_content(url):
   page = requests.get(url,headers={"Accept-Language":"en-US"})
   return BeautifulSoup(page.text,"html.parser")
soup = get_page_content(url)


# In[168]:


# In[169]:


district = []
companies = []
for a in soup.findAll('span',class_='badge badge-u'):
  for text in a:
    #print(text)
    companies.append(text)
  #print(re.findall(r'\d+',a.string))
  #data.append(re.findall(r'\d+',a.string))
for a in soup.find_all('a', href=True):
  if 'Quận' in a.text or 'Huyện' in a.text:
    district.append(a.text)
    


# In[170]:


company_density = pd.DataFrame(list(zip(district,companies)),
               columns =['District', 'Total Companies'])


# In[171]:





# In[172]:


company_density.replace('Quận','District',regex=True,inplace=True)
company_density.replace('Huyện','',regex=True,inplace=True)


# In[173]:


company_density.drop([5],inplace = True)


# In[174]:


company_density.reset_index(drop=True, inplace=True)


# In[175]:


i = 17
while i < 24:
  company_density.at[i,'District'] = company_density.at[i,'District'].replace('District ','')
  i+=1


# In[176]:





# In[177]:


i = 0
while i < 5:
  company_density.at[i,'District'] = company_density['District'][i] + ' District'
  i+=1
i = 17
while i < 24:
  company_density.at[i,'District'] = company_density['District'][i] + ' District'
  i+=1


# In[178]:


company_density['District'] = natsorted(company_density['District'])


# In[179]:


#Convert Vietnamese to English
i = 0 
for i in range(len(company_density['District'])):
  data = convert(company_density['District'][i])
  company_density.at[i,'District'] = data
  i+=1 


# In[180]:





# In[181]:


hcmcdf_full = hcmcdf_with_houseprice.merge(company_density,on='District',how='left')
hcmcdf_full.at[0,'Total Companies'] = company_density.at[0,'Total Companies']
hcmcdf_full.at[3,'Total Companies'] = company_density.at[1,'Total Companies']
hcmcdf_full.at[4,'Total Companies'] = company_density.at[2,'Total Companies']
hcmcdf_full.at[18,'Total Companies'] = company_density.at[3,'Total Companies']
hcmcdf_full.at[19,'Total Companies'] = company_density.at[4,'Total Companies']


# In[182]:


#Dữ liệu hoàn chỉnh
print(hcmcdf_full)


# In[183]:


#hcmcdf_full.to_csv('hcmcdf_full.csv')


# # Vẽ bản đồ của TPHCM 

# In[184]:


# get the coordinates of HCMC
address = 'Ho Chi Minh, VietNam'

geolocator = Nominatim(user_agent="Your_app-name")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Ho Chi Minh, VietNam {}, {}.'.format(latitude, longitude))


# In[185]:


# create map of Ho Chi Minh using latitude and longitude values
map_hcmc = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, neighborhood in zip(hcmc_df['Latitude'], hcmc_df['Longitude'], hcmc_df['District']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_hcmc)  
    
map_hcmc
map_hcmc.save("HCMMap.html")
st.header("HCMC MAP")
HtmlFile = open("HCMMap.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code,width=1000, height=1000)
# # Lấy dữ liệu địa điểm của từng Quận
# 

# In[186]:



CLIENT_ID = 'Q0UO3BYFEAJJRKEBAYDGA4BJ5DY2GPKTW053JZCANOXCPPQX'
CLIENT_SECRET = 'Q0OXCBE4XDUSEE0HLSONDUVAFO1XMRKDADAEXFAHNGOO40FU'
VERSION = '20180323'


# In[187]:


def getNearbyVenues(names, latitudes, longitudes, radius=1500, LIMIT=150):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        # print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        15
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['District', 
                  'District Latitude', 
                  'District Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[188]:


hcm_venues = getNearbyVenues(names=hcmc_df['District'],
                             latitudes=hcmc_df['Latitude'],
                             longitudes=hcmc_df['Longitude']
                            )


# In[189]:

st.text('HCM Venues Data')
hcm_venues


# In[193]:


#hcm_venues.to_csv('hcm_venues.csv')


# In[190]:


#Check những địa điểm là quán ăn Việt Nam
st.text('HCM Venues that are Vietnamese Restaurant')
hcm_venues[hcm_venues['Venue Category'] == 'Vietnamese Restaurant']


# In[191]:


#Count địa điểm theo quận để biết coi số lượng địa điểm của từng quận khác nhau như thế nàonào
hcm_venues_group = hcm_venues.groupby('District').count().reset_index()
#hcm_venues_group


# ## Methodology <a name="methodology"></a>

# Bây giờ em đã có Data chi tiết về những yếu tố muốn phân tích của từng Quận , và có cả danh sách những địa điểm kinh doanh của từng quận trong tphcm. Tiếp theo, em sẽ tiến hành phân tích sơ qua dữ liệu để xem mức độ tập trung của những địa điểm kinh doanh trong mỗi quận là như thế nào, bà trong những điểm kinh doanh đó thì có bao nhiêu địa điểm là quán ăn Việt Nam. Sau đó sẽ tiến hành Clustering 24 quận huyện để lọc ra những Quận có mật độ quán ăn Việt Nam cao dựa theo kết quả clustering để biết được mức độ cạnh tranh của các quận.  
# 
# Thuật toán em sử dụng để clustering là thuật toán <b> KMean <b> và KMode:
# 
# - <b> Về điểm chung : <b> 2 Thuật toán KMean và Kmode này, đều dựa vào việc phân chia dữ liệu thành các cụm khác nhau sao cho cùng một cụm có tính chất giống nhau => tập hợp các điểm ở gần nhau trong một không gian 
# - <b> Về điểm khác nhau : 
#     + KMeans : convert qualitative data về interget categorical data.Phụ thuộc vào cluster mean nên dễ bị ảnh hưởng bởi outliers. 
#     + KModes : Chuyên dùng để clustering categorical data. Vì sử dụng mode nên không có cluster mean, khó bị ảnh hưởng bởi outliners

# # data exploration analysis & visualization <a name="analysis"></a>

# In[192]:


print('There are {} uniques categories.'.format(len(hcm_venues['Venue Category'].unique())))


# In[194]:


#hcm_venues['Venue Category'].unique()


# In[195]:


# check if the results contain "Vietnamese Restaurant"
#"Vietnamese Restaurant" in hcm_venues['Venue Category'].unique()


# **Số địa điểm từng quận**

# In[196]:


ax = hcm_venues_group.sort_values(by="Venue", ascending=False).plot(x="District", y="Venue", kind="bar")
ax.set_ylabel("Number of venues")


# **Số loại địa điểm từng Quận**

# In[197]:


hcm_venues_category = (
    hcm_venues.groupby(['District','Venue Category'])
        .count().reset_index()[['District', 'Venue Category']]
            .groupby('District').count().reset_index()
)
# hcm_venues_group_cat
ax = hcm_venues_category.sort_values(by="Venue Category", ascending=False).plot(x="District", y="Venue Category", kind="bar")
ax.set_ylabel("Number of categories")


# **Tần số của các loại địa điểm**
# 

# In[198]:


most_venues = hcm_venues.groupby('Venue Category').count().sort_values(by="Venue", ascending=False)


# In[199]:


#most_venues.head(10)


# In[200]:


#most_venues.tail(10)


# **Phân tích từng Quận**

# Sử dụng One Hot Encoding để mã hóa những loại địa điểm 

# In[201]:


# one hot encoding
hcm_onehot = pd.get_dummies(hcm_venues[['Venue Category']], prefix="", prefix_sep="")

# add district column back to dataframe
hcm_onehot['District'] = hcm_venues['District'] 

# move district column to the first column
fixed_columns = [hcm_onehot.columns[-1]] + list(hcm_onehot.columns[:-1])
hcm_onehot = hcm_onehot[fixed_columns]

# group the rows by district and by taking the mean of the frequency of occurrence of each category
hcm_grouped = hcm_onehot.groupby('District').mean().reset_index()
#hcm_grouped.head()
# group rows by neighborhood order by the mean of the frequency of occurrence of each category


# **Top 10 Địa Điểm từng Quận **

# In[202]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['District']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
hcm_10 = pd.DataFrame(columns=columns)
hcm_10['District'] = hcm_grouped['District']

for ind in np.arange(hcm_grouped.shape[0]):
    hcm_10.iloc[ind, 1:] = return_most_common_venues(hcm_grouped.iloc[ind, :], num_top_venues)

#hcm_10

#TOP 10 VENUES CATEGORIES FOR EACH DISTRICT


# In[203]:


len(hcm_grouped[hcm_grouped["Vietnamese Restaurant"] > 0])


# **Lọc ra 1 DataFrame chỉ có Quán Ăn Việt Nam**
# 

# In[204]:


hcm_grouped_restaurant = hcm_grouped[["District", "Vietnamese Restaurant"]]



# In[205]:


# Use the K-Means clustering to do this but first we need to determine how many k we need to use. The *"elbow" method* helps to find a good k.
# try with 10 different values of k to find the best one
Ks = 10
distortions = []

hcm_restaurant_clustering = hcm_grouped_restaurant.drop('District', 1)

for k in range(1, Ks):

    # run k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(hcm_restaurant_clustering)

    # find the distortion w.r.t each k
    distortions.append(
        sum(np.min(cdist(hcm_restaurant_clustering, kmeans.cluster_centers_, 'euclidean'), axis=1))
        / hcm_restaurant_clustering.shape[0]
    )

plt.plot(range(1, Ks), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# ***=> "elbow" appears at k=3 & k = 4, will test both k = 3 and k = 4 ***

# In[206]:


nclusters = 3
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(hcm_restaurant_clustering)


# In[207]:


# set number of clusters
kclusters = 3

hcmc_clustering = hcm_grouped_restaurant.drop(["District"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(hcmc_clustering)

# check cluster labels generated for each row in the dataframe
#kmeans.labels_[0:10]


# In[208]:


# create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
hcmc_merged = hcm_grouped_restaurant.copy()

# add clustering labels
hcmc_merged["Cluster Labels"] = kmeans.labels_


# In[209]:


#hcmc_merged


# In[210]:


# merge hcmc_merged with hcmc_df to add latitude/longitude for each neighborhood
hcmc_merged_final = hcmc_merged.join(hcmc_df.set_index("District"), on="District")

 # check the last columns!


# In[211]:


# sort the results by Cluster Labels

hcmc_merged_final.sort_values(["Cluster Labels"], inplace=True)



# In[212]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(hcmc_merged_final['Latitude'], hcmc_merged_final['Longitude'], hcmc_merged_final['District'], hcmc_merged_final['Cluster Labels']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Có thể thấy cluster 2 là những quận trung tâm trong khi cluster 1 là những quận ở vùng ven . Có 2 outlier là Củ Chi và Cần Giờ, khá là xa và cũng không có mật độ dân số cao. Người dân chỉ sinh sống ở những khu dân cư cố định. Tuy nhiên chia như vầy cũng không đầy đủ lắm vì có những quận tuy không phải trung tâm nhưng lại rất phát triển , ví dụ như quận 7 , quận 4 lại xếp chung cluster với Bình Chánh, Hóc Môn. Có những quận lọt vào cluster trung tâm nhưng lại không phát triển bằng và cũng khá xa trung tâm như Tân Phú. 

# K = 3 thì model có vẻ chưa đem lại đầy đủ thông tin. 

# In[213]:


#hcmc_merged_final.loc[hcmc_merged['Cluster Labels'] == 3]


# In[214]:


#hcmc_merged_final.loc[hcmc_merged['Cluster Labels'] == 2]


# In[215]:


#hcmc_merged_final.loc[hcmc_merged['Cluster Labels'] == 1]


# In[216]:


#hcmc_merged_final.loc[hcmc_merged['Cluster Labels'] == 0]


# # Test with number of cluster = 4 

# In[217]:


nclusters = 4
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(hcm_restaurant_clustering)


# In[218]:


# set number of clusters
kclusters = 4

hcmc_clustering_4 = hcm_grouped_restaurant.drop(["District"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(hcmc_clustering_4)

# check cluster labels generated for each row in the dataframe
#kmeans.labels_[0:10]


# In[219]:


# create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
hcmc_merged_4 = hcm_grouped_restaurant.copy()

# add clustering labels
hcmc_merged_4["Cluster Labels"] = kmeans.labels_


# In[220]:


#hcmc_merged_4


# In[221]:


# merge hcmc_merged with hcmc_df to add latitude/longitude for each neighborhood
hcmc_merged_final_4 = hcmc_merged_4.join(hcmc_df.set_index("District"), on="District")

 # check the last columns!


# In[222]:


# sort the results by Cluster Labels

hcmc_merged_final_4.sort_values(["Cluster Labels"], inplace=True)
#hcmc_merged_final_4


# In[223]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(hcmc_merged_final_4['Latitude'], hcmc_merged_final_4['Longitude'], hcmc_merged_final_4['District'], hcmc_merged_final_4['Cluster Labels']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Có thể thấy là với k = 4 thì bây giờ có 3 cluster rõ rệt. Củ Chi và Cần Giờ vẫn là 2 outlier 

# Cluster 0 chính là những Quận phát triển và những Quận trung tâm, cluster 0 giờ đây đã xếp chung quận 1 với những quận như quận 4 và quận 7, thì điều này khá hợp lý. 

# Cluster 1 chính là những quận ở rìa thành phố, gần sân bay như quận 10 , Tân Bình, Phú Nhuận. Thì những quận này có mật độ dân số cao tuy nhiên lại xa trung tâm, gần sân bay và các doanh trại, căn cứ quân đội nên việc kinh doanh buôn bán, công ty nhà xưởng ,... không được đa dạng và nhiều như những quận ở cluster 0. Hầu hết chỉ tập trung ở 1 số khu vực nhất định như Sư Vạn Hạnh quận 10 , Phan Xích Long hay cách mạng tháng 8 . 
# 

# Cluster 2 là Củ Chi , mà Củ Chi hiện tại đang là một outlier nên chúng ta sẽ không xét tới Cluster này. Tiếp theo là Cluster 3, là những Quận/ Huyện ngoại thành như quận 2, quận 9 , Thủ Đức, Hóc Môn, quận 12... Những quận này là cách trung tâm xá kha và cũng mới phát triển thời gian gần đây nên hoạt động kinh tế, nhà hàng quán ăn và mật độ dân số cũng còn nhiều tiềm năng. Hầu như mật độ nhà hàng quán ăn cũng chưa quá đông đúc và nổi tiếng. 

print('Clustering Results by KMeans')
map_clusters.save("KMeans.html")
import streamlit.components.v1 as components

# >>> import plotly.express as px
# >>> fig = px.box(range(10))
# >>> fig.write_html('test.html')

st.header("KMeans ")

HtmlFile = open("KMeans.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code,width=1000, height=1000)
# In[224]:


hcmc_merged_final_4.loc[hcmc_merged_final_4['Cluster Labels'] == 2]


# In[225]:


hcmc_merged_final_4.loc[hcmc_merged_final_4['Cluster Labels'] == 3]


# In[226]:


hcmc_merged_final_4.loc[hcmc_merged_final_4['Cluster Labels'] == 1]


# In[227]:


hcmc_merged_final_4.loc[hcmc_merged_final_4['Cluster Labels'] == 0]


# **SO SÁNH VỚI THUẬT TOÁN KMODES**

# *Count Vietnamese Restaurant Frequency by District*

# In[228]:


df = hcm_venues[hcm_venues['Venue Category'] == 'Vietnamese Restaurant']


# In[229]:


df1 = df[['District','Venue Category']].groupby(['District']).agg('count')


# In[230]:


df1.reset_index(inplace=True)


# In[231]:


#df1


# Convert Quantitative data về categorical data

# In[232]:


df1.rename(columns={'Venue Category':'Venue_Marks'},inplace=True)


# In[233]:


district_list = hcmc_merged_final[['District']]


# In[234]:


data_kmode = district_list.merge(df1,on='District',how='left')


# *Fill NA = 00*

# In[235]:


data_kmode['Venue_Marks'] = data_kmode['Venue_Marks'].fillna(0)


# In[236]:


#data_kmode


# *Ranking Type from Venue Marks*

# In[237]:


#df1['Venue_Marks'].describe()


# In[238]:


data_kmode['Type'] = data_kmode.apply(lambda x: 'Low' if (x['Venue_Marks'] >= 1 and x['Venue_Marks'] < 3) 
                                   else ('Medium' if (x['Venue_Marks'] >= 3 and x['Venue_Marks'] < 5) 
                                         else ('High' if (x['Venue_Marks'] >= 5 and x['Venue_Marks'] < 9) 
                                               else 'Very High')), axis = 1)


# In[239]:


#data_kmode


# In[240]:


data_kmode.drop(columns=['Venue_Marks'],inplace=True)


# In[241]:


#data_kmode


# In[242]:





# In[243]:


# importing necessary libraries
import pandas as pd
import numpy as np
# !pip install kmodes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt



# In[244]:


# Elbow curve to find optimal K
cost = []
K = range(1,10)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
    kmode.fit_predict(data_kmode)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()


# *Elbow appears at k = 4*

# In[245]:


# Building the model with 4 clusters
kmode = KModes(n_clusters=4, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(data_kmode)



# In[246]:


clustersDf = pd.DataFrame(clusters)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([data_kmode, clustersDf], axis = 1)


# In[247]:


#combinedDf


# In[248]:


final_kmode = combinedDf.merge(hcmc_merged_final,on='District',how='left').drop(columns=['Cluster Labels'])


# In[249]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(final_kmode['Latitude'], final_kmode['Longitude'], final_kmode['District'], final_kmode['cluster_predicted']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Xét về mặt ý nghĩa thì phương pháp cluster bằng K Mode vẫn có ý nghĩa của nó, Cluster 0 là những quận nội thành còn cluster 2 là những quận nằm ở cách trung tâm thành phố 3 - 4 km. Cluster 1 là Quận Thủ Đức, nay đã lên thành phố Thủ Đức và cũng là trung tâm của thành phố Thủ Đức thì khu vực này cũng có tiềm năng để trở thành 1 cluster riêng biệt trong tương lai. Còn lại cluster 3 là khu vực những quận ngoại thành xa trung tâm thành phố như quận Bình Tân, Quận 6 và Quận 2, Cần Giờ.  

# Nếu dùng thuật toán K mode để clustering thì chúng ta sẽ thu nhận được thêm Cần Giờ và Củ Chi mà không phải loại đi vì outliers. Tuy nhiên lúc này Quận Thủ Đức lại đứng độc lập thành 1 cluster riêng biệt. Nhưng điều này là hợp lý vì tương lai Thủ Đức sẽ là 1 thành phố độc lập

# #** Examine other factors **#

# **Average Housing Price (AHP)**
print('Clustering results by KModes')
map_clusters.save("KModes.html")
st.header("KModes")
HtmlFile = open("KModes.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code,width=1000, height=1000)
# In[250]:




# In[251]:


df_housing_price['Average Housing Price (1M VND)/m2'].astype(str).astype(float).describe()


# In[252]:




# In[253]:





# In[254]:


hcmcdf_with_houseprice['Average Housing Price (1M VND)/m2'] = hcmcdf_with_houseprice['Average Housing Price (1M VND)/m2'].astype(str).astype(float)


# Range of Average Housing Price
# *   Low :  9.6 ≤ AHP < 70.25
# *   Medium :  70.25 ≤ AHP < 99.6 
# *   High :  99.6 ≤ AHP < 144 
# *   Very High :  144 ≤ AHP 
# 
# 
# 
# 
# 
# 

# In[255]:


hcmcdf_with_houseprice['AHP_Level'] = hcmcdf_with_houseprice.apply(lambda x: 'Low' if (x['Average Housing Price (1M VND)/m2'] >= 9.6 and x['Average Housing Price (1M VND)/m2'] < 70.25) 
                                   else ('Medium' if (x['Average Housing Price (1M VND)/m2'] >= 70.25 and x['Average Housing Price (1M VND)/m2'] < 99.6) 
                                         else ('High' if (x['Average Housing Price (1M VND)/m2'] >= 99.6 and x['Average Housing Price (1M VND)/m2'] < 144) 
                                               else 'Very High')), axis = 1)


# In[256]:


AHP_Result = hcmcdf_with_houseprice.merge(hcmc_merged_4,on='District',how = 'inner')


# In[257]:




# In[258]:


#AHP_Result[['District','Average Housing Price (1M VND)/m2','AHP_Level','Cluster Labels']]


# Ý Nghĩa của các cluster :
# 
# *   Cluster 0 : Quận trung tâm và có kinh tế phát triển, mức độ cạnh tranh cao.
# *   Cluster 1 : Những Quận có mức độ cạnh tranh tương đối.
# *   Cluster 3 : Những quận ngoại thành mới phát triển, có mức độ cạnh tranh còn thấp.
# 
# Cluster 0 là outlier nên loại bỏ ( Củ Chi ) 
# 
# 
# 
# 
# 
# 

# **population density**

# In[259]:


#AHP_Result


# In[260]:


#AHP_Result['Density'].astype(str).astype(float).describe()


# In[261]:



# In[262]:


AHP_Result['Density'] = AHP_Result['Density'].astype(str).astype(float)


# Range of Average Population Density
# *   Low :  7 ≤ APD < 3459.25
# *   Medium :  3459.25 ≤ APD < 22272.5
# *   High :  22272.5 ≤ APD < 36358 
# *   Very High :  36358 ≤ APD 
# 
# 
# 
# 
# 
# 

# In[263]:


AHP_Result['APD_Level'] = AHP_Result.apply(lambda x: 'Low' if (x['Density'] >= 7 and x['Density'] < 3459.25) 
                                   else ('Medium' if (x['Density'] >= 3459.25 and x['Density'] < 22272.5) 
                                         else ('High' if (x['Density'] >= 22272.5 and x['Density'] < 36358) 
                                               else 'Very High')), axis = 1)


# In[264]:


consideration = AHP_Result[['District','Average Housing Price (1M VND)/m2','AHP_Level','APD_Level','Cluster Labels']]


# In[265]:





# In[266]:


consideration = consideration.merge(company_density,on='District',how='left')
consideration.at[0,'Total Companies'] = company_density.at[0,'Total Companies']
consideration.at[3,'Total Companies'] = company_density.at[1,'Total Companies']
consideration.at[4,'Total Companies'] = company_density.at[2,'Total Companies']
consideration.at[18,'Total Companies'] = company_density.at[3,'Total Companies']
consideration.at[19,'Total Companies'] = company_density.at[4,'Total Companies']


# In[267]:

st.subheader("Results Data")
consideration

# In[268]:



# ## Results and Discussion <a name="results"></a>

# In[278]:


compare = pd.DataFrame(consideration.groupby(['AHP_Level','APD_Level'])['APD_Level'].count())
st.text(' Thống kê Average Housing Prive & Average Population Density')
compare

# Từ kết quả tóm tắt ở trên , ta chỉ muốn những quận có chi phí bất động sản vừa phải , hoặc "hời hơn" nếu so với mật độ dân số ( nghĩa là APD level cao hơn APH Level ), chứ không ai muốn những quận có chi phí bất động sản quá cao nhưng mật độ dân số thì lại thấp. Hoặc những quận có mật độ dân số cao , chi phí bất động sản cũng cao không kém thì cũng chẳng có lợi ích gì. 
# 
# Lọc ra được những cặp như sau : 
# 
# AHP Level Low - APD Level High : 1 Quận ( chi phí nhà thấp, mật độ dân số cao ) <br>
# AHP Level Medium - APD Level High : 1 Quận ( chi phí nhà tương đối, mật độ dân số cao ) <br>
# AHP Level Medium - APD Level Very High : 1 Quận ( chi phí nhà tương đối , mật độ dân số rất cao) 

# <b> Chi Phí nhà tương đối, mật độ dân số rất cao là Quận 4 <b>

# In[270]:

st.text('Quận có Chi Phí nhà tương đối, mật độ dân số rất cao là Quận 4')
consideration[(consideration['AHP_Level'] == 'Medium') & (consideration['APD_Level'] == 'Very High')]

# Nhắc lại ý nghĩa cluster : 
# 
# *   Cluster 0 : Có ít cạnh tranh.
# *   Cluster 1 : Có sự cạnh tranh tương đối.
# *   Cluster 2 : Có sự cạnh tranh cao .

# **Việc mở quán ăn ở quận 4, mặc dù có lợi thế là giá bất động sản ở mức giá tương đối và mật độ dân cư cao, tuy nhiên, mật độ công ty / nhà xưởng trong khu vực này lại quá thấp và mức cạnh tranh rất cao nên khá rủi ro khi kinh doanh tại đây.**

# <b> Chi Phí nhà tương đối, mật độ dân số cao là quận 8 <b>

# In[279]:

st.text('Quận có Chi Phí nhà tương đối, mật độ dân số cao là Quận 8 ')
consideration[(consideration['AHP_Level'] == 'Medium') & (consideration['APD_Level'] == 'High')]

# <b> Mặc dù quận 8 có giá bất động sản ở mức tương đối và mật độ dân cư cao, tuy nhiên vẫn không hấp dẫn do không có nhiều công ty và nhà xưởng hơn quận 4 là mấy, và mức cạnh tranh cũng rất cao <b>
# 

# <b> Quận có chi phí nhà thấp , mật độ dân số tương đối là quận 12 <b>

# In[280]:

st.text('Quận có chi phí nhà thấp , mật độ dân số tương đối là Quận 12 ')
consideration[(consideration['AHP_Level'] == 'Low') & (consideration['APD_Level'] == 'Medium')]


#  **việc mở 1 quán ăn Việt Nam ở quận 12 sẽ là tốt nhất vì có mật độ dân số tương đối, mật độ công ty , nhà xưởng vào mức cao trong khi giá bất động sản lại chỉ bằng 3/5 so với quận 4 và quận 8**

# => Vậy , lựa chọn tốt nhất là nên mở một quán ăn Việt Nam ở quận 12 vì chi phí bất động sản rẻ, mật đô dân cư, công ty & nhà máy xí nghiệp tương đối cao và mức độ canh tranh thấp.

# ## Conclusion <a name="conclusion"></a>

# Mục đích của Đồ Án này là tìm ra quận hợp lý nhất để mở một nhà hàng Việt Nam dựa trên các tiêu chí : Chi Phí cố định vừa phải & kinh tế ( chi phí bất động sản ), mật độ dân cư đông và số lượng công ty , nhà máy xí nghiệp nhiều để không chỉ kinh doanh cho dân địa phương mà còn cho công nhân của các nhà máy , nhân viên của các công ty do các đối tượng này có xu hướng ra ngoài ăn trưa, ghé ăn sáng ,... hoặc đi làm về ghé ăn nhậu. Sau khi tiến hành phân tích dữ liệu từ nhiều nguồn và xây dựng mô hình phân tích cụm thì quận hợp lý nhất để mở một nhà hàng Việt Nam là quận 12.

# Ngoài ra, Kết quả này là một sự dự đoán khách quan từ dữ liệu kết hợp với cả suy luận theo tình hình thực tế dựa trên kết quả của dữ liệu. Đương nhiên sẽ không thể tránh khỏi những thiếu sót như mật độ quán ăn chưa phản ánh đúng thực tế ( vì tình hình quán ăn của Việt Nam đôi khi không có trong dữ liệu của Foursquare, hoặc dữ liệu từ Foursquare dán nhãn sai, ví dụ như quán ăn cơm trưa văn phòng Việt Nam nhưng lại dán nhãn Cà phê,... Tuy nhiên ban đầu việc collect dữ liệu đã lấy ngẫu nhiên 100 địa điểm ở từng quận, nếu quận nào có mật độ nhà hàng Việt Nam nhiều đương nhiên kết quả sẽ ra nhiều nên cũng hạn chế được vấn đề này.     
