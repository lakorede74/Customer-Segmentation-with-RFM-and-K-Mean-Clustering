# 🧩Customer-Segmentation-with-RFM-and-K-Mean-Clustering

## Overview
 This project segments or groups customers based on based on Recency Frequency and Monetary (RFM) behaviour and further segments using K-Mean clustering. This will enables the company to know how various customers interacts with their brand and also optimize their business strategy.
>*This analysis only considered delivered orders*

 ## Dataset
 Data source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
 
 Description: This is a real commercial Dataset of orders from a Brazilian E-commerce (Olist), though it has been anonymized. It consists of 100k orders spanned from 2016 - 2018
 
 File name: Olist
 
 ## Objectives
 - Group customers based on share characterististics
 - segement customers in via RFM (Recency, Frequency and Monetary)
 - Segment using K-Mean Clustering Algorithm
 - Identify different segments of customers
 - Recommend improvement strategies


## Data Cleaning
---
    #Importing The necessary libraries
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    import numpy as np
    from sklearn import preprocessing
    import datetime as dt

    #Loading the Data 
    
    customer_df = pd.read_csv('./OList/olist_customers_dataset.csv')
    order_df = pd.read_csv('./OList/olist_orders_dataset.csv')
    products_df = pd.read_csv('./OList/olist_products_dataset.csv')
    order_items_df = pd.read_csv('./OList/olist_order_items_dataset.csv')
    payment_df = pd.read_csv('./OList/olist_order_payments_dataset.csv')

     data_df = pd.merge(customer_df, order_df,  on='customer_id')
     
     data_df = pd.merge(data_df, order_items_df, on='order_id')
     data_df1 = pd.merge(data_df, payment_df, on='order_id')

     #Check for Null 
     data_df1.isnull().sum()

---

# Out[1]:
```text

        customer_id                         0
        customer_unique_id                  0
        customer_zip_code_prefix            0
        customer_city                       0
        customer_state                      0
        order_id                            0
        order_status                        0
        order_purchase_timestamp            0
        order_approved_at                  15
        order_delivered_carrier_date     1245
        order_delivered_customer_date    2567
        order_estimated_delivery_date       0
        order_item_id                       0
        product_id                          0
        seller_id                           0
        shipping_limit_date                 0
        price                               0
        freight_value                       0
        payment_sequential                  0
        payment_type                        0
        payment_installments                0
        payment_value                       0
        dtype: int64

```

---
    #View data head
    data_df1.head(10)
---

# Out[2]: 
```text


                        customer_id                customer_unique_id  \
0  06b8999e2fba1a1fbc88172c00ba8bc7  861eff4711a542e4b93843c6dd7febb0   
1  18955e83d337fd6b2def6b18a428ac77  290c77bc529b7ac935b93aa66c333dc3   
2  4e7b3e00288586ebd08712fdd0374a03  060e732b5b29e8181a18229c7b0b2b5e   
3  b2b6027bc5c5109e529d4dc6358b12c3  259dac757896d24d7702b9acbbff3f3c   
4  4f2d8ab171c80ec8364f7c12e35b23ad  345ecd01c38d18a9036ed96c73b8d066   

   customer_zip_code_prefix          customer_city customer_state  \
0                     14409                 franca             SP   
1                      9790  sao bernardo do campo             SP   
2                      1151              sao paulo             SP   
3                      8775        mogi das cruzes             SP   
4                     13056               campinas             SP   

                           order_id order_status order_purchase_timestamp  \
0  00e7ee1b050b8499577073aeb2a297a1    delivered      2017-05-16 15:05:35   
1  29150127e6685892b6eab3eec79f59c7    delivered      2018-01-12 20:48:24   
2  b2059ed67ce144a36e2aa97d2c9e9ad2    delivered      2018-05-19 16:07:45   
3  951670f92359f4fe4a63112aa7306eba    delivered      2018-03-13 16:06:38   
4  6b7d50bd145f6fc7f33cebabd7e49d0f    delivered      2018-07-29 09:51:30   

     order_approved_at order_delivered_carrier_date  \
0  2017-05-16 15:22:12          2017-05-23 10:47:57   
1  2018-01-12 20:58:32          2018-01-15 17:14:59   
2  2018-05-20 16:19:10          2018-06-11 14:31:00   
3  2018-03-13 17:29:19          2018-03-27 23:22:42   
4  2018-07-29 10:10:09          2018-07-30 15:16:00   

  order_delivered_customer_date order_estimated_delivery_date  order_item_id  \
0           2017-05-25 10:35:35           2017-06-05 00:00:00              1   
1           2018-01-29 12:41:19           2018-02-06 00:00:00              1   
2           2018-06-14 17:58:51           2018-06-13 00:00:00              1   
3           2018-03-28 16:04:25           2018-04-10 00:00:00              1   
4           2018-08-09 20:55:48           2018-08-15 00:00:00              1   

                         product_id                         seller_id  \
0  a9516a079e37a9c9c36b9b78b10169e8  7c67e1448b00f6e969d365cea6b010ab   
1  4aa6014eceb682077f9dc4bffebc05b0  b8bc237ba3788b23da09c0f1f3a3288c   
2  bd07b66896d6f1494f5b86251848ced7  7c67e1448b00f6e969d365cea6b010ab   
3  a5647c44af977b148e0a3a4751a09e2e  7c67e1448b00f6e969d365cea6b010ab   
4  9391a573abe00141c56e38d84d7d5b3b  4a3ca9315b744ce9f8e9374361493884   

   shipping_limit_date   price  freight_value  payment_sequential  \
0  2017-05-22 15:22:12  124.99          21.88                   1   
1  2018-01-18 20:58:32  289.00          46.48                   1   
2  2018-06-05 16:19:10  139.94          17.79                   1   
3  2018-03-27 16:31:16  149.94          23.36                   1   
4  2018-07-31 10:10:09  230.00          22.25                   1   

  payment_type  payment_installments  payment_value  
0  credit_card                     2         146.87  
1  credit_card                     8         335.48  
2  credit_card                     7         157.73  
3  credit_card                     1         173.30  
4  credit_card                     8         252.25  
```


##
