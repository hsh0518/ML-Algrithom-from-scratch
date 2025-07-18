trainingdata = [
  {'user_id': '123', 'age': 25, 'gender': 'F'},
  {'user_id': '234', 'age': 30, 'gender': 'M'},
  ...
]
schema = [
  {"name": "age"}, {"name": "gender"}
]
返回一个 list of lists，每一行：
[user_id, feature1, feature2, ..., label]

def dataprocessing(trainingdata, schema, labels):
    output = []
    feature_keys = [f['name'] for f in schema]

    for row in trainingdata:
        user_id = row['user_id']
        features = [row.get(key, None) for key in feature_keys]
        label = labels.get(user_id, None)
        output.append([user_id] + features + [label])

    return output
#test
trainingdata = [
    {'user_id': '123', 'age': 25, 'gender': 'F'},
    {'user_id': '234', 'age': 30, 'gender': 'M'},
]

schema = [{'name': 'age'}, {'name': 'gender'}]

labels = {'123': True, '234': False}

print(dataprocessing(trainingdata, schema, labels))
# Output:
# [
#   ['123', 25, 'F', True],
#   ['234', 30, 'M', False]
# ]
