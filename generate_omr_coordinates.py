import json

# Student ID and Paper Code coordinates
student_id_x = [38, 76, 117, 158, 198, 238, 278]
paper_code_x = [750, 788, 828, 868, 908, 949, 989]
letters_y = [335, 363, 390, 418, 447, 474, 502, 530, 558, 586]
id_radius = 10

# Questions coordinates
question_sections = {
    "1-13": {
        "x": [75, 104, 134, 163, 192, 222],
        "y": [681, 718, 755, 792, 829, 865, 902, 939, 976, 1012, 1049, 1086, 1123],
        "start": 1,
        "end": 13
    },
    "14-29": {
        "x": [332, 361, 391, 420, 449, 479],
        "y": [681, 718, 755, 792, 829, 865, 902, 939, 976, 1012, 1049, 1086, 1123, 1159, 1196, 1233],
        "start": 14,
        "end": 29
    },
    "30-49": {
        "x": [589, 619, 648, 677, 707, 736],
        "y": [681, 718, 755, 792, 829, 865, 902, 939, 976, 1012, 1049, 1086, 1123, 1159, 1196, 1233, 1269, 1306, 1343, 1380],
        "start": 30,
        "end": 49
    },
    "50-70": {
        "x": [846, 876, 905, 935, 964, 994],
        "y": [681, 718, 755, 792, 829, 865, 902, 939, 976, 1012, 1049, 1086, 1123, 1159, 1196, 1233, 1269, 1306, 1343, 1380, 1417],
        "start": 50,
        "end": 70
    }
}
question_radius = 9

output = {
    "student_id": {},
    "paper_code": {},
    "questions": {}
}

# Generate student_id
for i, x in enumerate(student_id_x):
    key = f"letter_{i+1}"
    output["student_id"][key] = []
    for y in letters_y:
        output["student_id"][key].append({"x": x, "y": y, "r": id_radius})

# Generate paper_code
for i, x in enumerate(paper_code_x):
    key = f"letter_{i+1}"
    output["paper_code"][key] = []
    for y in letters_y:
        output["paper_code"][key].append({"x": x, "y": y, "r": id_radius})

# Generate questions
for section in question_sections.values():
    for i, q in enumerate(range(section["start"], section["start"] + len(section["y"]))):
        key = f"question_{q}"
        output["questions"][key] = []
        y = section["y"][i]
        for x in section["x"]:
            output["questions"][key].append({"x": x, "y": y, "r": question_radius})

# Write to file
with open("assets/omr_coordinates.json", "w") as f:
    json.dump(output, f, indent=2)

print("omr_coordinates.json generated successfully.")
