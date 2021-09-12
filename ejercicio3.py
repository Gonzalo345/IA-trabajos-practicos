# -*- coding: utf-8 -*-
import numpy as np

print("\n---- Ejercicio 3 ----")
print("Construir un índice para identificadores de usuarios, id2idx e idx2id\n")

'''
El objetivo es construir un índice para identificadores de usuarios,
es decir id2idx e idx2id. Para ello crear una clase, donde el
índice se genere en el constructor. Armar métodos get_users_id 
y get_users_idx.

Identificadores de usuarios : users_id = [15, 12, 14, 10, 1, 2, 1]
Índice de usuarios : users_id = [0, 1, 2, 3, 4, 5, 4]

id2idx =  [-1     4     5    -1    -1    -1     -1    -1    -1    -1     3     -1      1    -1     2     0]
          [ 0     1     2     3     4     5      6     7     8     9    10     11     12    13    14    15]

id2idx[15] -> 0 ; id2idx[12] -> 1 ; id2idx[3] -> -1
idx2id[0] -> 15 ; idx2id[4] -> 1

'''


class Usuario:
    instance = None

    def __new__(self, id2id):
        if Usuario.instance is None:
            print("__new__ usuario creado")
            Usuario.instance = super(Usuario, self).__new__(self)
            return Usuario.instance
        else:
            return Usuario.instance

    def __init__(self, id2id):
        print("__init__")
        self.id2id = id2id
        self.idx = np.full(shape=np.max(id2id) + 1, fill_value=-1)
        print("valores", id2id)
        # u, indices = np.unique(id2id, return_index=True)
        mylist = list(dict.fromkeys(id2id))
        print("valores", mylist)
        print(type(mylist))
        j = 0
        for i in mylist:
            self.idx[i] = j
            j = j + 1
        print("Idx \n", self.idx)

    def get_user_id(self, id2):
        return (self.id2id[id2])

    def get_user_idx(self, id2):
        return (self.idx[id2])


id2id = np.array([15, 12, 14, 10, 1, 2, 1])
user_1 = Usuario(id2id)

print("Get user id  2 =", user_1.get_user_id(2))
print("Get user idx 4 =", user_1.get_user_idx(5))

print(hex(id(user_1)))
