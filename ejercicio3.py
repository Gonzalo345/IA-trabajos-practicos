import numpy as np

print("\n---- Ejercicio 3 ----")
print("Construir un índice para identificadores de usuarios, id2idx e idx2id\n")

'''
Indexing 

El objetivo es construir un índice para identificadores de usuarios,es decir id2idx e idx2id. Para ello crear una clase, 
donde el índice se genere en el constructor. Armar métodos get_users_id y get_users_idx.

Identificadores de usuarios : users_id = [15, 12, 14, 10, 1, 2, 1]
Índice de usuarios :          users_id = [0,   1,  2,  3, 4, 5, 4]

id2idx =  [-1     4     5    -1    -1    -1     -1    -1    -1    -1     3     -1      1    -1     2     0]
          [ 0     1     2     3     4     5      6     7     8     9    10     11     12    13    14    15]

id2idx[15] -> 0 ; id2idx[12] -> 1 ; id2idx[3] -> -1
idx2id[0] -> 15 ; idx2id[4] -> 1

'''


class Usuario:
    instance = None

    def __init__(self, usuario_id):
        print("__init__")
        self.usuario_id = usuario_id
        # Creo un array con el maximo valor de user_id relleno con -1
        self.usuario_idx = np.full(shape=np.max(usuario_id) + 1, fill_value=-1)
        print("Usuarios_id          ", usuario_id)  # Usuarios id
        print("Usuario_idx vacío    ", self.usuario_idx)
        # Busco los valores unicos
        unique_id, indices = np.unique(usuario_id, return_index=True)

        # Invierto el orden
        # unique_id = -np.sort(-unique_id)
        print("Usuarios_id_unicos   ", unique_id, "Indices", indices)  # Usuarios id
        # Completo el vector de indices con la información de usuarios
        self.usuario_idx[unique_id] = indices
        print("Usuario_idx completo ", self.usuario_idx)

    def id2idx(self, user_id):
        return self.usuario_idx[user_id]

    def idx2id(self, user_idx):
        return self.usuario_id[user_idx]


usuario_id = np.array([15, 12, 14, 10, 1, 2, 1])
user_1 = Usuario(usuario_id)

print("id2idx[15] -> ", user_1.id2idx(15), "; id2idx[12] -> ", user_1.id2idx(12), "; id2idx[3] -> ", user_1.id2idx(3))
print("idx2id[0] -> ", user_1.idx2id(0), "; idx2id[4] -> ", user_1.idx2id(4))

