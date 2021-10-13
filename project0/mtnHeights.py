mountains = {'Mount Everest': 8848,
             'Makalu': 8485,
             'Nanga Parbat': 8126,
             'Annapurna': 8091,
             'Broad Peak': 8051
             }

print("These are some of the tallest mountains in the world: ")
for name in mountains.keys():
    print(name)

print("\nHere are their elevations (meters): ")
for elevation in mountains.values():
    print(elevation)

print("\nMountains matched with and their elevations:")
for name, elevation in mountains.items():
    print('%s is %s meters tall' % (name, elevation))

