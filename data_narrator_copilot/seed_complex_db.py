import psycopg2
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Database Connection (Replace YOUR_ACTUAL_PASSWORD)
DB_URI = "dbname='data_narrator' user='postgres' password='postgres' host='localhost' port='5432'"

def create_complex_schema(cursor):
    print("🗑️ Dropping old tables if they exist...")
    cursor.execute("""
        DROP TABLE IF EXISTS sales CASCADE;
        DROP TABLE IF EXISTS employees CASCADE;
        DROP TABLE IF EXISTS customers CASCADE;
        DROP TABLE IF EXISTS products CASCADE;
        DROP TABLE IF EXISTS departments CASCADE;
        DROP TABLE IF EXISTS regions CASCADE;
    """)

    print("🏗️ Creating complex relational schema...")
    cursor.execute("""
        CREATE TABLE regions (
            region_id SERIAL PRIMARY KEY,
            region_name VARCHAR(50) UNIQUE NOT NULL
        );
        CREATE TABLE departments (
            dept_id SERIAL PRIMARY KEY,
            dept_name VARCHAR(50) UNIQUE NOT NULL
        );
        CREATE TABLE products (
            product_id SERIAL PRIMARY KEY,
            product_name VARCHAR(100) NOT NULL,
            category VARCHAR(50) NOT NULL,
            price DECIMAL(10, 2) NOT NULL,
            cost DECIMAL(10, 2) NOT NULL
        );
        CREATE TABLE customers (
            customer_id SERIAL PRIMARY KEY,
            company_name VARCHAR(100) NOT NULL,
            industry VARCHAR(50) NOT NULL,
            region_id INT REFERENCES regions(region_id)
        );
        CREATE TABLE employees (
            emp_id SERIAL PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            dept_id INT REFERENCES departments(dept_id),
            region_id INT REFERENCES regions(region_id),
            hire_date DATE NOT NULL,
            salary DECIMAL(10, 2) NOT NULL
        );
        CREATE TABLE sales (
            sale_id SERIAL PRIMARY KEY,
            product_id INT REFERENCES products(product_id),
            customer_id INT REFERENCES customers(customer_id),
            emp_id INT REFERENCES employees(emp_id),
            sale_date DATE NOT NULL,
            quantity INT NOT NULL,
            total_amount DECIMAL(15, 2) NOT NULL
        );
    """)

def insert_dummy_data(cursor):
    print("💉 Injecting realistic enterprise data...")
    
    # 1. Regions
    regions = ['North America', 'EMEA', 'APAC', 'LATAM']
    for r in regions:
        cursor.execute("INSERT INTO regions (region_name) VALUES (%s)", (r,))
    
    # 2. Departments
    departments = ['Enterprise Sales', 'Cloud Infrastructure', 'AI Solutions', 'Customer Success']
    for d in departments:
        cursor.execute("INSERT INTO departments (dept_name) VALUES (%s)", (d,))

    # 3. Products (50 items)
    categories = ['Software License', 'Cloud Storage', 'Hardware Server', 'Consulting', 'Security']
    for _ in range(50):
        name = f"{fake.company()} {fake.word().capitalize()} System"
        category = random.choice(categories)
        cost = round(random.uniform(100, 5000), 2)
        price = round(cost * random.uniform(1.3, 2.5), 2) # 30% to 150% markup
        cursor.execute("INSERT INTO products (product_name, category, price, cost) VALUES (%s, %s, %s, %s)", 
                       (name, category, price, cost))

    # 4. Customers (200 companies)
    industries = ['Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Technology']
    for _ in range(200):
        cursor.execute("INSERT INTO customers (company_name, industry, region_id) VALUES (%s, %s, %s)",
                       (fake.company(), random.choice(industries), random.randint(1, 4)))

    # 5. Employees (100 reps)
    for _ in range(100):
        hire_date = fake.date_between(start_date='-5y', end_date='today')
        cursor.execute("INSERT INTO employees (full_name, dept_id, region_id, hire_date, salary) VALUES (%s, %s, %s, %s, %s)",
                       (fake.name(), random.randint(1, 4), random.randint(1, 4), hire_date, round(random.uniform(60000, 150000), 2)))

    # 6. Sales (5000 transactions over the last 2 years)
    print("⏳ Generating 5,000 sales transactions (this may take a few seconds)...")
    for _ in range(5000):
        product_id = random.randint(1, 50)
        quantity = random.randint(1, 100)
        
        # Fetch actual price to calculate total_amount accurately
        cursor.execute("SELECT price FROM products WHERE product_id = %s", (product_id,))
        price = cursor.fetchone()[0]
        total_amount = price * quantity
        
        sale_date = fake.date_between(start_date='-2y', end_date='today')
        
        cursor.execute("""
            INSERT INTO sales (product_id, customer_id, emp_id, sale_date, quantity, total_amount) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (product_id, random.randint(1, 200), random.randint(1, 100), sale_date, quantity, total_amount))

def main():
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()
        
        create_complex_schema(cursor)
        insert_dummy_data(cursor)
        
        conn.commit()
        print("\n✅ Complex Database successfully seeded with thousands of rows!")
    except Exception as e:
        print(f"❌ Database error: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    main()