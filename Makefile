#.DEFAULT_GOAL := generate #choose the default goal instead of it being the first one
.PHONY: all clean #tells make that these goals are not files but some other thing

# compiler options, debug and performance
cppDebug = g++ -std=c++17 -Wall -Wextra -Wpedantic
cppPerf = g++ -std=c++17 -O3
cppCompiler = ${cppDebug} #the version used

# list of binaries to clean as well
binaries = ADD file names


all:
	@echo "This does nothing. See Makefile"

tally: tally.f90
	${cppCompiler} $< -o $@

# Create any object files needed
%.o: %.f90
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
