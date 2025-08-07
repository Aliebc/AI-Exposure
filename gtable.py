import pandas as pd
import os

os.makedirs('./output', exist_ok=True)

EXCEL_DIR = "../data/db_29_3_excel/"
EXCEL_ABIL = "../data/db_29_3_excel/Abilities.xlsx"

df = pd.read_excel(EXCEL_ABIL)
print(df.head())

ABILITIES = df['Element Name'].unique()

abilities_table = pd.DataFrame({
    'Element ID': range(1, len(ABILITIES) + 1),
    'Element Name': ABILITIES,
})

abilities_table['Element ID'] = abilities_table['Element ID'].astype(str)


TRUNCATE = 30
abilities_table = pd.concat([
    abilities_table.iloc[i:i+TRUNCATE].reset_index(drop=True)
    for i in range(0, len(abilities_table), TRUNCATE)
], axis=1)
    
print(abilities_table)
# Latex pretty table
latex_table = abilities_table.to_latex(index=False, escape=False)
with open('./output/abilities_table.tex', 'w') as f:
    latex_table = latex_table.replace('NaN', '---')  # Replace NaN with a placeholder
    f.write(latex_table)
    
OCCUPATIONS = df['Title'].unique()
occupations_table = pd.DataFrame({
    'Occupation ID': range(1, len(OCCUPATIONS) + 1),
    'Occupation Name': OCCUPATIONS,
})
occupations_table = occupations_table.sample(40, random_state=42).reset_index(drop=True)

occupations_table['Occupation ID'] = occupations_table['Occupation ID'].astype(str)

latex_occupations_table = occupations_table.to_latex(index=False, escape=False, 
                                                     column_format=R'lp{10cm}')
with open('./output/occupations_table.tex', 'w') as f:
    latex_occupations_table = latex_occupations_table.replace('NaN', '---')  # Replace NaN with a placeholder
    f.write(latex_occupations_table)

print(occupations_table)

EXCEL_DWA = os.path.join(EXCEL_DIR, 'DWA Reference.xlsx')
df2 = pd.read_excel(EXCEL_DWA)

print(df2.head())

DWAs = df2['DWA Title'].drop_duplicates()

dwa_table = pd.DataFrame({
    'DWA ID': range(1, len(DWAs) + 1),
    'DWA Title': DWAs,
})

dwa_table['DWA ID'] = dwa_table['DWA ID'].astype(str)

dwa_table_sample = dwa_table.sample(40, random_state=42).reset_index(drop=True)
latex_dwa_table = dwa_table_sample.to_latex(index=False, escape=False,
                                            column_format=R'lp{10cm}')
with open('./output/dwa_table.tex', 'w') as f:
    latex_dwa_table = latex_dwa_table.replace('NaN', '---')  # Replace NaN with a placeholder
    f.write(latex_dwa_table)

print(dwa_table)