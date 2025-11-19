import re
import pandas as pd
import numpy as np
from pathlib import Path

# rutas absolutas (reproducible)
IN_PATH = "/Users/cristophercoronavelasco/Desktop/Ciencia de Datos/Taller de Modelado de Datos/data/dataset_uncleaned.csv"
OUT_DIR = Path("/Users/cristophercoronavelasco/Desktop/Ciencia de Datos/Taller de Modelado de Datos/results")

# --- helpers sencillos ---

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    # limpieza basica de strings y placeholders raros (sin applymap)
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        s = df[c].astype(str).str.strip().str.strip('_, "').str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
        df[c] = s
    df = df.replace(['!@9#%8', '#F%$D@*&8'], np.nan)
    return df

def convert_credit_history_age(x):
    # "X Years Y Months" -> total_meses
    if pd.isna(x): return np.nan
    s = str(x).lower().replace('_',' ').replace('-',' ')
    m_years = re.search(r'(\d+)\s*year', s)
    m_months = re.search(r'(\d+)\s*month', s)
    years = int(m_years.group(1)) if m_years else 0
    months = int(m_months.group(1)) if m_months else 0
    total = years*12 + months
    return float(total) if total >= 0 else np.nan

def hex_to_int_safe(x):
    # hex a entero, toleramos 'cus_' y '0x'
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace('cus_','')
    if s.startswith('0x'): s = s[2:]
    try:
        return int(s, 16)
    except Exception:
        return np.nan

MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12
}
def month_to_num(x):
    # mes texto -> num 1..12
    if pd.isna(x): return np.nan
    return MONTH_MAP.get(str(x).strip().lower(), np.nan)

def clean_type_of_loan(cell):
    # prestamos: separar, normalizar, sin dups
    if pd.isna(cell): return np.nan
    s = str(cell).lower().replace(' and ', ',')
    s = re.sub(r'[|/]+', ',', s)
    s = re.sub(r',+', ',', s)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    alias = {
        'credit builder loan':'Credit Builder Loan',
        'credit-builder loan':'Credit Builder Loan',
        'home equity loan':'Home Equity Loan',
        'auto loan':'Auto Loan',
        'personal loan':'Personal Loan',
        'student loan':'Student Loan',
        'debt consolidation loan':'Debt Consolidation Loan',
        'mortgage loan':'Mortgage Loan',
        'payday loan':'Payday Loan'
    }
    canon, seen = [], set()
    for p in parts:
        name = alias.get(p, p.title())
        if name not in seen:
            seen.add(name)
            canon.append(name)
    return '; '.join(canon) if canon else np.nan

def clean_ssn(x):
    # SSN: letras comunes -> digitos y formato XXX-XX-XXXX
    if pd.isna(x): return np.nan
    s = str(x).translate(str.maketrans({'O':'0','o':'0','I':'1','l':'1','S':'5','B':'8'}))
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) != 9: return np.nan
    return f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"

def clip_quantiles(s: pd.Series, q_low=0.01, q_hi=0.99, floor=None, ceil=None):
    # recorte por cuantiles (outliers)
    lo, hi = s.quantile(q_low), s.quantile(q_hi)
    s = s.clip(lower=lo, upper=hi)
    if floor is not None: s = s.clip(lower=floor)
    if ceil is not None: s = s.clip(upper=ceil)
    return s

# --- pipeline Lite+ (simple) ---

# leemos datos
df = pd.read_csv(IN_PATH, low_memory=False)

# limpieza de texto
df = clean_text_columns(df)

# columnas especificas
if 'Credit_History_Age' in df.columns:
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history_age)

if 'ID' in df.columns:
    df['ID'] = df['ID'].apply(hex_to_int_safe).astype('Int64')

if 'Customer_ID' in df.columns:
    df['Customer_ID'] = df['Customer_ID'].apply(hex_to_int_safe).astype('Int64')

if 'Month' in df.columns:
    df['Month'] = df['Month'].apply(month_to_num).astype('Int64')

if 'Type_of_Loan' in df.columns:
    df['Type_of_Loan'] = df['Type_of_Loan'].apply(clean_type_of_loan)

if 'SSN' in df.columns:
    df['SSN'] = df['SSN'].apply(clean_ssn)

# numericos
num_cols = [
    'Age','Annual_Income','Monthly_Inhand_Salary',
    'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate',
    'Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
    'Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt',
    'Credit_Utilization_Ratio','Credit_History_Age','Total_EMI_per_month',
    'Amount_invested_monthly','Monthly_Balance'
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# orden por cliente/mes
if {'Customer_ID','Month'}.issubset(df.columns):
    df = df.sort_values(['Customer_ID','Month'])

### parches para bajar nans

    #1. imputación categórica basica por cliente a global
for c in ['Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour','Occupation','Name']:
    if c in df.columns:
        #por cliente primero
        df[c] = df.groupby('Customer_ID')[c].ffill().bfill()
        #si aun falta: moda global
        moda = df[c].dropna().mode()
        if not moda.empty:
            df[c] = df[c].fillna(moda.iloc[0])

# 2. Type_of_Loan: deja 'Unknown' si sigue vacio
if 'Type_of_Loan' in df.columns:
    df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Unknown')

# 3. Credit_History_Age: rellenar con mediana por cliente y luego global
if 'Credit_History_Age' in df.columns:
    med_cus = df.groupby('Customer_ID')['Credit_History_Age'].transform('median')
    df['Credit_History_Age'] = df['Credit_History_Age'].fillna(med_cus)
    df['Credit_History_Age'] = df['Credit_History_Age'].fillna(df['Credit_History_Age'].median())

# 4. Numericos: usar transform('mean') por cliente y si falta mediana global
num_cols = [
    'Age','Annual_Income','Monthly_Inhand_Salary',
    'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate',
    'Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
    'Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt',
    'Credit_Utilization_Ratio','Credit_History_Age','Total_EMI_per_month',
    'Amount_invested_monthly','Monthly_Balance'
]
for c in num_cols:
    if c in df.columns:
        mean_cus = df.groupby('Customer_ID')[c].transform('mean')
        df[c] = df[c].fillna(mean_cus)
        df[c] = df[c].fillna(df[c].median())

# Imputación basica (texto/sueldos) por cliente
for c in ['Name','Occupation','Monthly_Inhand_Salary','Annual_Income','Credit_Mix',
          'Payment_of_Min_Amount','Payment_Behaviour']:
    if c in df.columns:
        df[c] = df.groupby('Customer_ID')[c].ffill().bfill()

# Imputación numerica: usar transform (no apply) para evitar indices raros
for c in num_cols:
    if c in df.columns:
        grp_mean = df.groupby('Customer_ID')[c].transform('mean')  # mean por cliente
        df[c] = df[c].fillna(grp_mean)
        df[c] = df[c].fillna(df[c].median())  # si aun falta, mediana global

# reglas de rango basicas
if 'Age' in df.columns:
    df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Credit_Utilization_Ratio' in df.columns:
    df['Credit_Utilization_Ratio'] = df['Credit_Utilization_Ratio'].clip(0, 100)

if 'Interest_Rate' in df.columns:
    df['Interest_Rate'] = df['Interest_Rate'].clip(1, 50)

# outliers: recorte simple
for c in ['Num_Credit_Inquiries','Num_of_Delayed_Payment','Delay_from_due_date',
          'Total_EMI_per_month','Monthly_Inhand_Salary','Amount_invested_monthly',
          'Monthly_Balance','Annual_Income','Outstanding_Debt']:
    if c in df.columns:
        floor = 0 if c in ['Num_Credit_Inquiries','Num_of_Delayed_Payment','Delay_from_due_date',
                           'Total_EMI_per_month','Monthly_Inhand_Salary','Amount_invested_monthly'] else None
        df[c] = clip_quantiles(df[c], 0.01, 0.99, floor=floor)

# guardar
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = OUT_DIR / "dataset_clean.csv"
df.to_csv(out_file, index=False)
print(f"OK -> {out_file}")
