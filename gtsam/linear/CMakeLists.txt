# Install headers
file(GLOB linear_headers "*.h")
if(NOT GTSAM_WITH_CUSPARSE)
    list(REMOVE_ITEM linear_headers "${CMAKE_CURRENT_SOURCE_DIR}/CuSparseSolver.h")
endif()
install(FILES ${linear_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/linear)

# Build tests
add_subdirectory(tests)
