import pandas as pd

dataset = pd.DataFrame({
    'test_score':      [35, 40, 28, 45, 33, 38, 29, 42, 31, 36],
    'writing_skills':  [30, 50, 25, 60, 28, 45, 22, 65, 33, 48],
    'reading_skills':  [32, 42, 20, 55, 30, 40, 18, 58, 28, 43],
    'attendance':      [60, 65, 55, 70, 50, 68, 58, 75, 62, 66],
    'study_hours':     [1, 2, 1, 3, 2, 2, 1, 3, 1, 2],
    'pass':            ['No','No','No','Yes','No','No','No','Yes','No','No']
})

print(dataset)