CC=nvcc

# Compiler flags
ifeq ($(OS), Windows_NT)
	ifeq ($(RELEASE),TRUE)
		-Xcompiler /Wall -s -O3 -std=c++17
	else
		-Xcompiler /Wall -g -std=c++17
	endif
else
	ifeq ($(RELEASE),TRUE)
		-Xcompiler -Wall -Xcompiler -Wextra -s -O3 -std=c++17
	else
		-Xcompiler -Wall -Xcompiler -Wextra -g -std=c++17
	endif
endif

all : windows linux

windows : obj/main.obj obj/device_query.obj
	$(CC) $(CFLAGS) -o device_query.exe $^

linux : obj/main.o obj/device_query.o
	$(CC) $(CFLAGS) -o device_query.out $^