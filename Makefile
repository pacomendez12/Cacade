CC=g++ -g
OPENCVLIB=`pkg-config --cflags --libs opencv`
#FLAGS=-std=c++11 -fopenmp
FLAGS=-std=c++11
SRC=src/classifier/cascade_data.cpp \
	src/classifier/classifier.cpp \
	src/classifier/feature_evaluator.cpp \
	src/classifier/group_rectangles.cpp
OBJS=src/classifier/cascade_data.o \
	 src/classifier/classifier.o \
	src/classifier/feature_evaluator.o \
	src/classifier/group_rectangles.o


all:
	$(CC) -o cascade src/classifier/main.cpp $(SRC) -Iinclude $(OPENCVLIB) $(FLAGS)

clean:
	rm cascade
