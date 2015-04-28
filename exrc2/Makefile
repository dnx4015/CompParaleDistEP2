CC=nvcc
CFLAGS=-arch=compute_11
SRCS=$(wildcard *.cu)
OUTS=$(SRCS:%.cu=%.out)

all: $(OUTS)

%.out: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm *.out
