games = ['Poker', 'Scrabble', 'Uno']
print('My favorite games are:')
for game in games:
    print(game)

new_game = input('\nWhat\'s your favorite game? ')

games.append(new_game)

print('\nOur favorite games are:')
for game in games:
    print(game)
