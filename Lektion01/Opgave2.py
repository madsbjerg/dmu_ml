
forfattere = ["Andersen", "Blixen", "Epictetus", "Aristotle"]

for f in forfattere:
    print(f)

forfattere.append("Mossen")

for f in forfattere:
    print(f)

length = len(forfattere)

print(length)

forfattere.pop(1)

forfattere.reverse()

for f in forfattere:
    print(f)