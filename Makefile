# Compiler
CXX = g++

# Include and Library Flags
PKG_CONFIG = `pkg-config --cflags --libs opencv4`
LIBS = -lportaudio -lsndfile

# Source and Output
SRC_DIR = src
SRC_FILES = $(SRC_DIR)/BabySounds.cpp
OUTPUT = BabySounds

# Default Target
all: $(OUTPUT)

# Compilation Rules
$(OUTPUT): $(SRC_DIR)
	$(CXX) $(CXXFlags) $(SRC_FILES) -o $(OUTPUT) $(PKG_CONFIG) $(LIBS)

# Clean Rule
clean:
	rm -f $(OUTPUT)