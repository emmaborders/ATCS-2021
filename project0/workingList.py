careers = ['doctor', 'teacher', 'chef', 'firefighter']

print(careers.index('teacher'))
print('teacher' in careers)
print()

careers.append('astronaut')
careers.insert(0, 'lawyer')

for career in careers:
    print(career)

