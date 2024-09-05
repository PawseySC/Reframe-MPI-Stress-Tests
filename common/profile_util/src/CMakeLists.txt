# set the source dir

set(PU_SOURCES
    mem_util.cpp
    thread_affinity_util.cpp
    time_util.cpp
    profile_util.cpp
)
if (PU_ENABLE_SHARED_LIB)
    add_library(profile_util SHARED ${PU_SOURCES})
else()
    add_library(profile_util STATIC ${PU_SOURCES})
endif()
target_compile_definitions(profile_util PUBLIC ${PU_DEFINES})
if (PU_CXX_FLAGS)
    target_compile_options(profile_util PUBLIC ${PU_CXX_FLAGS})
endif()
set_target_properties(profile_util PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
install(TARGETS profile_util 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

