
.SUFFIXES:
.SUFFIXES:  .o .C

SRCS :=$(wildcard *.C)
OBJS_SRC=$(SRCS:.C=.o)
OBJS := $(notdir $(OBJS_SRC))

CXX=icpc
LDFLAGS=-qopenmp
#CXXFLAGS=-xmic-avx512 -O3 -fp-model consistent -qopenmp -DOPT=float
CXXFLAGS=-xmic-avx512 -O3 -fp-model precise -fp-model source -qopenmp -DOPT=float


BIN=evc

$(BIN): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(BIN)

.C.o:
	$(CXX) -o $@ $(CXXFLAGS) -c $<

clean:
	rm -f *.o  $(BIN)
