import json

file = json.load(open("scene_validation_annotations_20170908.json"))

var_label = open("val_label.txt","w")
for i in range(len(file)):
    dict = file[i]
    var_label.write("{} {}\n".format(dict['image_id'],dict['label_id']))
var_label.close()
