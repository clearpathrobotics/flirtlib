SET(feature_SRCS
    InterestPoint.cpp
    MultiScaleDetector.cpp
    RangeDetector.cpp
    NormalDetector.cpp
    NormalEdgeDetector.cpp
    NormalBlobDetector.cpp
    CurvatureDetector.cpp
    ShapeContext.cpp
    BetaGrid.cpp
    AbstractFeatureSetMatcher.cpp
    RansacFeatureSetMatcher.cpp
    RansacMultiFeatureSetMatcher.cpp
)


ADD_LIBRARY(flirtlib_feature SHARED ${feature_SRCS})
TARGET_LINK_LIBRARIES(flirtlib_feature flirtlib_utils flirtlib_geometry flirtlib_sensors ${Boost_SYSTEM_LIBRARY} ${Boost_TIMER_LIBRARY} ${Boost_SERIALIZATION_LIBRARY})

install(TARGETS flirtlib_feature
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/flirtlib/feature DESTINATION include/flirtlib)
