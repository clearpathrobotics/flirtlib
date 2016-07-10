SET(sensors_SRCS
    AbstractReading.cpp
    LaserReading.cpp
)

ADD_LIBRARY(flirtlib_sensors SHARED ${sensors_SRCS})
TARGET_LINK_LIBRARIES(flirtlib_sensors flirtlib_geometry)

install(TARGETS flirtlib_sensors
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/flirtlib/sensors DESTINATION include/flirtlib)
