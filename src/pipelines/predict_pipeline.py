import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            trans_path=r'C:\Users\syada11\Stores_sales_prediction\artifacts\preprocessor.pkl'
            model_path=r'C:\Users\syada11\Stores_sales_prediction\artifacts\best_model.pkl'
            transformer=load_object(file_path=trans_path)
            model=load_object(file_path=model_path)
            data_scaled = transformer.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        Item_Identifier,
        Item_Weight,
        Item_Fat_Content,
        Item_Visibility,
        Item_Type,
        Item_MRP,
        Outlet_Establishment_Year,
        Outlet_Size,
        Outlet_Location_Type,
        Outlet_Type,
        Outlet_Age):
       
        # Mapping the form inputs to expected model inputs
        self.Item_Identifier = Item_Identifier
        self.Item_Weight = Item_Weight
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Visibility = Item_Visibility
        self.Item_Type = Item_Type
        self.Item_MRP = Item_MRP
        self.Outlet_Establishment_Year = Outlet_Establishment_Year
        self.Outlet_Size =Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type
        self.Outlet_Age = Outlet_Age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dic = {
                "Item_Identifier" : [self.Item_Identifier],
                "Item_Weight": [self.Item_Weight],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Visibility": [self.Item_Visibility],
                "Item_Type": [self.Item_Type],
                "Item_MRP": [self.Item_MRP],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year],
                "Outlet_Size" : [self.Outlet_Size],
                "Outlet_Location_Type" : [self.Outlet_Location_Type],
                "Outlet_Type" : [self.Outlet_Type],
                "Outlet_Age": [self.Outlet_Age]
            }
            return pd.DataFrame(custom_data_input_dic)
        except Exception as e:
            raise CustomException(e, sys)
