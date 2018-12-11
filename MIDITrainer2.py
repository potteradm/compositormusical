import glob

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Activation
from music21 import converter, instrument, note, chord
import numpy
from keras.utils import np_utils

parseNota =dict()
parseInt=dict()

def readNotes(pasta):
    notes = []

    for file in glob.glob(pasta+"/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes,sorted(set(notes))

def criarModelo(entrada,nsaida):
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(entrada.shape[1], entrada.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.4))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(nsaida))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model



def preencherMapa(nsets):
    global parseNota
    global parseInt
    for i in range(len(nsets)):
        parseInt[i] = nsets[i];
        parseNota[nsets[i]]=i;


def parse(v):
    global parseNota
    global parseInt
    if type(v) is str:
        return parseNota[v];
    else:
        return parseInt[v];


def preparaDados(dados,sequencia):
    entrada=[]
    saida=[]
    for i in range(0, len(dados) - sequencia):
        sequence_in = dados[i:i + sequencia] #pega os 100 elementos
        sequence_out = dados[i + sequencia] #pega o proximo elemento
        entrada.append([parse(v) for v in sequence_in])
        saida.append(parse(sequence_out))

    entrada=numpy.reshape(entrada,(entrada.__len__(),sequencia,1))  #parametriza para a entrada da LSTM
    entrada = entrada / float(parseNota.__len__())
    saida = np_utils.to_categorical(saida)
    return entrada,saida



def executar():
    local=input("digite a pasta do dataset")
    notas,ordclassnotas=readNotes(local)#carrega o dataset de notas
    print(len(notas))
    preencherMapa(ordclassnotas)#preenche o mapa de conversao
    entrada,saida=preparaDados(notas,100)
    print(len(entrada))
    model=criarModelo(entrada,ordclassnotas.__len__())

    filepath ="weights-"+local+"-{epoch:02d}-{loss:.4f}.hdf5"

    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(entrada, saida, epochs=200, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    executar()
