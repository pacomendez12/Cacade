CC=g++
OPENCVLIB=`pkg-config --cflags --libs opencv`
SRC=src/classifier/cascade_data.cpp
OBJS=src/classifier/cascade_data.o


all:
	$(CC) -o cascade src/classifier/main.cpp $(SRC) -Iinclude $(OPENCVLIB)

clean:
	rm cascade
