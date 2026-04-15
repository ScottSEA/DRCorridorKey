cmake_minimum_required(VERSION 3.18)
project(CorridorKeyOFX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────

# libtorch — download from https://pytorch.org/get-started/locally/
# Set Torch_DIR to the libtorch/share/cmake/Torch directory
find_package(Torch REQUIRED)

# OpenCV — optional, used for matte cleanup (connected components)
find_package(OpenCV COMPONENTS core imgproc QUIET)

# OpenFX SDK — set OFX_SDK_DIR to the root of the OpenFX SDK
# Download from https://github.com/AcademySoftwareFoundation/openfx
if(NOT DEFINED OFX_SDK_DIR)
    message(FATAL_ERROR
        "OFX_SDK_DIR is not set.  Download the OpenFX SDK from:\n"
        "  https://github.com/AcademySoftwareFoundation/openfx\n"
        "Then set -DOFX_SDK_DIR=/path/to/openfx"
    )
endif()

# ── Sources ───────────────────────────────────────────────────────────

set(PLUGIN_SOURCES
    src/plugin_main.cpp
    src/corridorkey_effect.cpp
    src/corridorkey_effect.h
    src/color.h
    src/preprocess.h
    src/postprocess.h
)

# Build as a shared library (.ofx is just a .dll/.so/.dylib)
add_library(CorridorKeyOFX SHARED ${PLUGIN_SOURCES})

# ── Include paths ─────────────────────────────────────────────────────

target_include_directories(CorridorKeyOFX PRIVATE
    ${OFX_SDK_DIR}/include          # OFX headers
    ${OFX_SDK_DIR}/Support/include  # OFX support library headers
    src/
)

# ── Link ──────────────────────────────────────────────────────────────

target_link_libraries(CorridorKeyOFX PRIVATE
    ${TORCH_LIBRARIES}
)

if(OpenCV_FOUND)
    target_link_libraries(CorridorKeyOFX PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(CorridorKeyOFX PRIVATE HAS_OPENCV=1)
endif()

# ── Output name ───────────────────────────────────────────────────────

# OFX plugins must have the .ofx extension
set_target_properties(CorridorKeyOFX PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    OUTPUT_NAME "CorridorKey"
)

# ── Install (bundle layout) ──────────────────────────────────────────
# OFX plugins are distributed as bundles:
#   CorridorKey.ofx.bundle/
#     Contents/
#       Win64/CorridorKey.ofx       (Windows)
#       Linux-x86-64/CorridorKey.ofx (Linux)
#       MacOS/CorridorKey.ofx       (macOS)

if(WIN32)
    set(OFX_ARCH "Win64")
elseif(APPLE)
    set(OFX_ARCH "MacOS")
else()
    set(OFX_ARCH "Linux-x86-64")
endif()

install(TARGETS CorridorKeyOFX
    LIBRARY DESTINATION "CorridorKey.ofx.bundle/Contents/${OFX_ARCH}"
    RUNTIME DESTINATION "CorridorKey.ofx.bundle/Contents/${OFX_ARCH}"
)

# Install the TorchScript model alongside the plugin
install(FILES "${CMAKE_SOURCE_DIR}/corridorkey.pt"
    DESTINATION "CorridorKey.ofx.bundle/Contents/Resources"
    OPTIONAL  # Model may not be present during development
)
