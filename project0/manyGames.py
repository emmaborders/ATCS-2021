games = ['Poker', 'Scrabble', 'Uno']
print('My favorite games are:')
for game in games:
    print(game)

print()

new_game = ''
while new_game != 'quit':
    new_game = input('Enter a game you like or type \'quit\': ')
    if new_game != 'quit':
        games.append(new_game)

print('\nOur favorite games are:')
for game in games:
    print(game)

