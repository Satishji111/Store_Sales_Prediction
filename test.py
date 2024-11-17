from src.utils import load_object

# Load the preprocessor
preprocessor = load_object(r'C:\Users\syada11\Stores_sales_prediction\artifacts\preprocessor.pkl')

# Get categories from OrdinalEncoder in the ColumnTransformer
ordinal_encoder = preprocessor.named_transformers_['ordi'].named_steps['encoder']
print("Categories in OrdinalEncoder:", ordinal_encoder.categories_)
