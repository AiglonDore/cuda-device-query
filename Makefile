WCC=nvcc
LCC=nvcc

# Compiler flags
ifeq ($(OS), Windows_NT)
	ifeq ($(RELEASE),TRUE)
		CFLAGS= -O3 -std=c++17
	else
		CFLAGS=-g -std=c++17
	endif
	LCC=wsl nvcc
	DEL=del
	SEP=\\
else
	ifeq ($(RELEASE),TRUE)
		CFLAGS=-Xcompiler -s -Xcompiler -Wall -Xcompiler -Wextra -O3 -std=c++17
	else
		CFLAGS=-Xcompiler -Wall -Xcompiler -Wextra -g -std=c++17
	endif
	DEL=rm
	SEP=/
endif

all : windows linux

windows : obj/main.obj obj/device_query.obj obj/exn.obj obj/query.obj
	$(WCC) $(CFLAGS) -o bin/device_query.exe $^

linux : obj/main.o obj/device_query.o obj/exn.o obj/query.o
	$(LCC) $(CFLAGS) -o bin/device_query.out $^

obj/main.obj : src/main.cpp header/device_query.cuh header/exn.h
	$(WCC) $(CFLAGS) -c -o $@ $<

obj/device_query.obj : src/device_query.cu header/device_query.cuh header/exn.h
	$(WCC) $(CFLAGS) -c -o $@ $<

obj/exn.obj : src/exn.cpp header/exn.h
	$(WCC) $(CFLAGS) -c -o $@ $<

obj/query.obj : src/query.cpp header/query.h header/device_query.cuh header/exn.h
	$(WCC) $(CFLAGS) -c -o $@ $<

obj/main.o : src/main.cpp header/device_query.cuh header/exn.h
	$(LCC) $(CFLAGS) -c -o $@ $<

obj/device_query.o : src/device_query.cu header/device_query.cuh header/exn.h
	$(LCC) $(CFLAGS) -c -o $@ $<

obj/exn.o : src/exn.cpp header/exn.h
	$(LCC) $(CFLAGS) -c -o $@ $<

obj/query.o : src/query.cpp header/query.h header/device_query.cuh header/exn.h
	$(LCC) $(CFLAGS) -c -o $@ $<

clean :
	$(DEL) obj$(SEP)*.obj
	$(DEL) obj$(SEP)*.o
	$(DEL) bin$(SEP)*.exe
	$(DEL) bin$(SEP)*.out
	$(DEL) bin$(SEP)*.pdb
	$(DEL) bin$(SEP)*.exp
	$(DEL) bin$(SEP)*.lib
	$(DEL) *.pdb