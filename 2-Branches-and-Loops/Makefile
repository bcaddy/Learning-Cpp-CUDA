#.DEFAULT_GOAL := generate #choose the default goal instead of it being the first one
.PHONY: all clean #tells make that these goals are not files but some other thing

# compiler options, debug and performance
cppDebug = g++ -std=c++17 -Wall -Wextra -Wpedantic
cppPerf = g++ -std=c++17 -O3
cppCompiler = ${cppDebug} #the version used

# list of binaries to clean as well
binaries = for-loops while-loops if switch immediate-if


all:
	@echo "This does nothing. See Makefile"

for-loops.exe: for-loops.cpp
	${cppCompiler} $< -o $@

while-loops.exe: while-loops.cpp
	${cppCompiler} $< -o $@

if.exe: if.cpp
	${cppCompiler} $< -o $@

switch.exe: switch.cpp
	${cppCompiler} $< -o $@

immediate-if.exe: immediate-if.cpp
	${cppCompiler} $< -o $@

# Create any object files needed
# %.o: %.cpp
# 	${cppCompiler} -c $<


clean:
	@echo "Cleaning up..."
	@rm -f $(binaries)
	@rm -f *.mod
	@rm -f *o
	@rm -f *.exe
	@rm -rf *.dSYM
	@rm -f a.out
	@echo "Done"
