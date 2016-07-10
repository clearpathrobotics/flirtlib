SET(sensorstream_SRCS
    CarmenLog.cpp
    LogSensorStream.cpp
)

SET(sensorstream_LIBS
    flirtlib_sensors
    flirtlib_geometry
)

ADD_LIBRARY(flirtlib_sensorstream SHARED ${sensorstream_SRCS})
TARGET_LINK_LIBRARIES(flirtlib_sensorstream ${sensorstream_LIBS})

install(TARGETS flirtlib_sensorstream
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/flirtlib/sensorstream DESTINATION include/flirtlib)
