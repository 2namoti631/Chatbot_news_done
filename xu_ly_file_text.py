import re 
import csv
import pandas as pd

# mở file để đọc và file csv để ghi
with open('train_5500label.text') as file, open('train5500.csv','w', newline= '') as csvfile:
    # tạo đối tượng writer csv 
    writer = csv.writer(csvfile)
    # viết header cho file csv 
    writer.writerow(['Classes', 'Definition','Questions'])

    # đọc từng dòng và xử lý file 
    for line in file: 
        # tách chữ ra 3 cột 
        match = re.match(r'^([A-Z]+):(\w+)\s(.+)$', line)
        if match:
            classes =  match.group(1)
            definition =  match.group(2)
            questions = match.group(3)

            # viết  dòng vào file csv 
            writer.writerow([classes, definition, questions])

df= pd.read_csv('train5500.csv', encoding='latin1')

# đổi tên các phần tử trong cột classes
df['Classes'] = df['Classes'].replace({
    'DESC': 'DESCRIPTION',
    'ENTY': 'ENTITY',
    'NUM' : 'NUMERIC',
    'HUM' : 'HUMAN' , 
    'ABBR':  'ABBREVIATION',
    'LOC' : 'LOCATION'
})
# đổi tên các phần tử trong cột definition
df.loc[(df['Classes'] == 'ENTITY') & (df['Definition'] == 'termeq'), 'Definition'] = 'term'
df.loc[(df['Classes'] == 'ENTITY') & (df['Definition'] == 'veh'), 'Definition'] = 'vehicle'
df.loc[(df['Classes'] == 'ENTITY') & (df['Definition'] == 'other'), 'Definition'] = 'other entity'
df.loc[(df['Classes'] == 'ENTITY') & (df['Definition'] == 'cremat'), 'Definition'] = 'creative'

df.loc[(df['Classes'] == 'DESCRIPTION') & (df['Definition'] == 'def'), 'Definition'] = 'definition'
df.loc[(df['Classes'] == 'DESCRIPTION') & (df['Definition'] == 'desc'), 'Definition'] = 'description of st'

df.loc[(df['Classes'] == 'HUMAN') & (df['Definition'] == 'desc'), 'Definition'] = 'description human'
df.loc[(df['Classes'] == 'HUMAN') & (df['Definition'] == 'gr'), 'Definition'] = 'group'

df.loc[(df['Classes'] == 'LOCATION') & (df['Definition'] == 'other'), 'Definition'] = 'other loc'

df.loc[(df['Classes'] == 'NUMERIC') & (df['Definition'] == 'dist'), 'Definition'] = 'distance'
df.loc[(df['Classes'] == 'NUMERIC') & (df['Definition'] == 'other'), 'Definition'] = 'other num'
df.loc[(df['Classes'] == 'NUMERIC') & (df['Definition'] == 'perce'), 'Definition'] = 'percent'

df.to_csv('train5500.csv', index=False)
