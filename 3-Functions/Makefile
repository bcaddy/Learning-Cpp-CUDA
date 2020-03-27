#.DEFAULT_GOAL := generate #choose the default goal instead of it being the first one
.PHONY: all clean #tells make that these goals are not files but some other thing

# compiler options, debug and performance
cppDebug = g++ -std=c++17 -Wall -Wextra -Wpedantic
cppPerf = g++ -std=c++17 -O3
cppCompiler = ${cppDebug} #the version used

all:
	@echo "This does nothing. See Makefile"

gravity.exe: gravity.o grav_force.o
	${cppCompiler} $? -o $@

PrintShapes.exe: PrintShapes.o PrintShapesUtility.o
	${cppCompiler} $? -o $@

%.o: %.cpp
	${cppCompiler} -c $<


clean:
	@echo "Cleaning up..."
	@rm -f $(binaries)
	@rm -f *.mod
	@rm -f *o
	@rm -f *.exe
	@rm -rf *.dSYM
	@rm -f a.out
	@echo "Done"
