letters = ['?', '!', '_', 'e', 't', 'a', 'n', 'o', 'i', 's', 'r', 'h', 'l',
           'u', 'd', 'c', 'm', 'f', 'p', 'k', 'g', 'y', 'b', 'w', '<', '>',
           'v', 'N', '.', "'", 'x', 'j', '$', '-', 'q', 'z', '&', '0', '1',
           '9', '3', '#', '2', '8', '5', '\\', '7', '6', '/', '4', '*']

le_word_to_id = list(zip(letters, range(len(letters))))
le_id_to_word = list(zip(range(len(letters)), letters))
