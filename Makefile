#*******************************************************************************************************/
#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
#*******************************************************************************************************/

# libs
BENCHMARK_PATH=../sys/benchmark
MKLML_PATH=../sys/mklml

# objects
OBJS= s8u8s32benchmark.o

# flags
CPPFLAGS= -O3 -std=c++11 -I$(BENCHMARK_PATH)/include -I$(MKLML_PATH)/include 
LINKFLAGS= -L$(BENCHMARK_PATH)/lib -L$(MKLML_PATH)/lib -lbenchmark -lmklml 

# final output
BIN=bin

$(BIN): $(OBJS)
	g++ $(LINKFLAGS) -o $@ $^

clean:
	rm -rf $(OBJS) $(BIN)

%.o:%.cpp
	g++ $(CPPFLAGS) -I. -c -o $@ $<
