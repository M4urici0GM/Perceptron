# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\Mauricio\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7846.88\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\Mauricio\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7846.88\bin\cmake\win\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Mauricio\CLionProjects\Perceptron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Perceptron.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Perceptron.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Perceptron.dir/flags.make

CMakeFiles/Perceptron.dir/main.cpp.obj: CMakeFiles/Perceptron.dir/flags.make
CMakeFiles/Perceptron.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Perceptron.dir/main.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Perceptron.dir\main.cpp.obj -c C:\Users\Mauricio\CLionProjects\Perceptron\main.cpp

CMakeFiles/Perceptron.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Perceptron.dir/main.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Mauricio\CLionProjects\Perceptron\main.cpp > CMakeFiles\Perceptron.dir\main.cpp.i

CMakeFiles/Perceptron.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Perceptron.dir/main.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\Mauricio\CLionProjects\Perceptron\main.cpp -o CMakeFiles\Perceptron.dir\main.cpp.s

CMakeFiles/Perceptron.dir/src/Perceptron.cpp.obj: CMakeFiles/Perceptron.dir/flags.make
CMakeFiles/Perceptron.dir/src/Perceptron.cpp.obj: ../src/Perceptron.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Perceptron.dir/src/Perceptron.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Perceptron.dir\src\Perceptron.cpp.obj -c C:\Users\Mauricio\CLionProjects\Perceptron\src\Perceptron.cpp

CMakeFiles/Perceptron.dir/src/Perceptron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Perceptron.dir/src/Perceptron.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Mauricio\CLionProjects\Perceptron\src\Perceptron.cpp > CMakeFiles\Perceptron.dir\src\Perceptron.cpp.i

CMakeFiles/Perceptron.dir/src/Perceptron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Perceptron.dir/src/Perceptron.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\Mauricio\CLionProjects\Perceptron\src\Perceptron.cpp -o CMakeFiles\Perceptron.dir\src\Perceptron.cpp.s

CMakeFiles/Perceptron.dir/src/Utils.cpp.obj: CMakeFiles/Perceptron.dir/flags.make
CMakeFiles/Perceptron.dir/src/Utils.cpp.obj: ../src/Utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Perceptron.dir/src/Utils.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Perceptron.dir\src\Utils.cpp.obj -c C:\Users\Mauricio\CLionProjects\Perceptron\src\Utils.cpp

CMakeFiles/Perceptron.dir/src/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Perceptron.dir/src/Utils.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Mauricio\CLionProjects\Perceptron\src\Utils.cpp > CMakeFiles\Perceptron.dir\src\Utils.cpp.i

CMakeFiles/Perceptron.dir/src/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Perceptron.dir/src/Utils.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\Mauricio\CLionProjects\Perceptron\src\Utils.cpp -o CMakeFiles\Perceptron.dir\src\Utils.cpp.s

# Object files for target Perceptron
Perceptron_OBJECTS = \
"CMakeFiles/Perceptron.dir/main.cpp.obj" \
"CMakeFiles/Perceptron.dir/src/Perceptron.cpp.obj" \
"CMakeFiles/Perceptron.dir/src/Utils.cpp.obj"

# External object files for target Perceptron
Perceptron_EXTERNAL_OBJECTS =

Perceptron.exe: CMakeFiles/Perceptron.dir/main.cpp.obj
Perceptron.exe: CMakeFiles/Perceptron.dir/src/Perceptron.cpp.obj
Perceptron.exe: CMakeFiles/Perceptron.dir/src/Utils.cpp.obj
Perceptron.exe: CMakeFiles/Perceptron.dir/build.make
Perceptron.exe: CMakeFiles/Perceptron.dir/linklibs.rsp
Perceptron.exe: CMakeFiles/Perceptron.dir/objects1.rsp
Perceptron.exe: CMakeFiles/Perceptron.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable Perceptron.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Perceptron.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Perceptron.dir/build: Perceptron.exe

.PHONY : CMakeFiles/Perceptron.dir/build

CMakeFiles/Perceptron.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Perceptron.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Perceptron.dir/clean

CMakeFiles/Perceptron.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\Mauricio\CLionProjects\Perceptron C:\Users\Mauricio\CLionProjects\Perceptron C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug C:\Users\Mauricio\CLionProjects\Perceptron\cmake-build-debug\CMakeFiles\Perceptron.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Perceptron.dir/depend

