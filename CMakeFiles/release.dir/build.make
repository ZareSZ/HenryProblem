# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/reza/projects/HenryProblem

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/reza/projects/HenryProblem

# Utility rule file for release.

# Include the progress variables for this target.
include CMakeFiles/release.dir/progress.make

CMakeFiles/release:
	$(CMAKE_COMMAND) -E cmake_progress_report /home/reza/projects/HenryProblem/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Switch CMAKE_BUILD_TYPE to Release"
	/usr/bin/cmake -DCMAKE_BUILD_TYPE=Release /home/reza/projects/HenryProblem
	/usr/bin/cmake --build /home/reza/projects/HenryProblem --target all

release: CMakeFiles/release
release: CMakeFiles/release.dir/build.make
.PHONY : release

# Rule to build all files generated by this target.
CMakeFiles/release.dir/build: release
.PHONY : CMakeFiles/release.dir/build

CMakeFiles/release.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/release.dir/cmake_clean.cmake
.PHONY : CMakeFiles/release.dir/clean

CMakeFiles/release.dir/depend:
	cd /home/reza/projects/HenryProblem && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/reza/projects/HenryProblem /home/reza/projects/HenryProblem /home/reza/projects/HenryProblem /home/reza/projects/HenryProblem /home/reza/projects/HenryProblem/CMakeFiles/release.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/release.dir/depend

