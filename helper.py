import json

def get_amount(data):
    """Calculates the cart amount as a categorical variable given the JSON data
    for a specific data entry.

    :param data: The JSON data string to parse."""
    data = json.loads(data)
    products = data.get("CartProduct", {"all": []})

    # Make sure we get all products in the cart.
    if "all" in products: products = products["all"]
    else : products = [products]

    amount = 0.0

    for p in products:
        try: amount += float(p["productPrice"]) * float(p["productQuantity"])
        except: pass
        
        return amount

def get_date(data):
    """Calculates the cart amount as a categorical variable given the JSON data
    for a specific data entry.

    :param data: The JSON data string to parse."""
    data = json.loads(data)
    dates = data.get("ReceiptData", {"orderDate": []})
    
    # Make sure we get all products in the cart.
    return dates['orderDate']