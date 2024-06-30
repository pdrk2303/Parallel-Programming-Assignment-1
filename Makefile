CC=g++
PTHREAD_FLAGS=-pthread
OPENMP_FLAGS=-fopenmp

all: a1_pthreads a1_openmp

a1_pthreads: a1_pthreads.cpp
	$(CC) $(PTHREAD_FLAGS) $< -o $@

a1_openmp: a1_openmp.cpp
	$(CC) $(OPENMP_FLAGS) $< -o $@

pthreads:
	./a1_pthreads

openmp:
	./a1_openmp

.PHONY: clean

clean:
	rm -f a1_pthreads a1_openmp
