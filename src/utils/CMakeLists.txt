SET(utils_SRCS
    Regression.cpp
    SimplePeakFinder.cpp
    SimpleMinMaxPeakFinder.cpp
    PoseEstimation.cpp
)

ADD_LIBRARY(flirtlib_utils SHARED ${utils_SRCS})

TARGET_LINK_LIBRARIES(flirtlib_utils flirtlib_geometry)

install(TARGETS flirtlib_utils
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/flirtlib/utils DESTINATION include/flirtlib)
