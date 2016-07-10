SET(geometry_SRCS
    point.cpp
)

ADD_LIBRARY(flirtlib_geometry SHARED ${geometry_SRCS})
TARGET_LINK_LIBRARIES(flirtlib_geometry ${Boost_SERIALIZATION_LIBRARY})

install(TARGETS flirtlib_geometry
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/flirtlib/geometry DESTINATION include/flirtlib)
