def crowd_test(people):
    if len(people) >= 3:
        print('This room is crowded!')


names = ['Sara', 'John', 'Rachel', 'Alex']
crowd_test(names)

names.remove('Sara')
names.remove('Alex')
crowd_test(names)


