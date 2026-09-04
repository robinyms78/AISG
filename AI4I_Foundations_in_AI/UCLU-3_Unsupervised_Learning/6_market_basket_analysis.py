import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def identify_customers(path: str):
    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(sum(df.Frequency))

    return df

def get_transaction(path: str, df: pd.DataFrame):
    tr = pd.read_csv(path)
    print(tr.head())
    tf_f = tr[tr['CustomerID'].isin(df.CustomerID)]
    print(tf_f.head())
    print(sum(tf_f.shape))

    tf_f['Description'] = tf_f['Description'].str.strip()
    tf_f['InvoiceNo'] = tf_f['InvoiceNo'].astype(str)
    print(tf_f.head())

    basket = (tf_f.groupby(['InvoiceNo','Description'])
    ['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    basket.drop('POSTAGE', inplace=True, axis=1)
    print(basket.head())

    return basket

def encode_units(x):
     if x <= 0:
         return 0
     if x >= 1:
         return 1

def get_association_rules(basket):
    # Binary encode the basket data
    basket_sets = basket.map(encode_units)
    print(basket_sets.head())

    # Create frequent itemsets using the Apriori algorithm
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
    print(frequent_itemsets.head())

    # Generate the association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print(rules.head())

    # Filter the rules based on lift and confidence thresholds
    filtered_rules = rules[ (rules['lift'] >= 6) &
        (rules['confidence'] >= 0.8) ].sort_values(['lift','confidence'], ascending=False)
    print(filtered_rules.head())

def main(path: str, processed_data_path: str):
    df = identify_customers(path)
    basket = get_transaction(processed_data_path, df)
    get_association_rules(basket)


if __name__ == "__main__":
    path = "eda_outputs/small_customers.csv"
    processed_data_path = "Dataset/processed_data.csv"
    main(path, processed_data_path)