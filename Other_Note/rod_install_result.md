Starting >>> rosidl_generator_cpp
Finished <<< rcl_logging_spdlog [47.9s]
Finished <<< class_loader [48.7s]
Starting >>> pluginlib
Finished <<< libyaml_vendor [53.6s]
Starting >>> rcl_yaml_param_parser
Finished <<< rosidl_typesupport_introspection_cpp [35.0s]
Finished <<< rmw [58.0s]
Starting >>> rosidl_typesupport_connext_cpp
Starting >>> rosidl_typesupport_fastrtps_cpp
Starting >>> rmw_connext_shared_cpp
--- stderr: rosidl_typesupport_fastrtps_cpp
CMake Error at CMakeLists.txt:26 (find_package):
  Could not find a package configuration file provided by "fastrtps" with any
  of the following names:

    fastrtpsConfig.cmake
    fastrtps-config.cmake

  Add the installation prefix of "fastrtps" to CMAKE_PREFIX_PATH or set
  "fastrtps_DIR" to a directory containing one of the above files.  If
  "fastrtps" provides a separate development package or SDK, be sure it has
  been installed.


---
Failed   <<< rosidl_typesupport_fastrtps_cpp [12.5s, exited with code 1]
Aborted  <<< rosidl_typesupport_connext_cpp [12.5s]
Aborted  <<< rmw_connext_shared_cpp [13.6s]
Aborted  <<< rcl_yaml_param_parser [18.1s]
Aborted  <<< pluginlib [39.3s]
Aborted  <<< rosidl_generator_cpp [44.5s]
Aborted  <<< foonathan_memory_vendor [2min 57s]
Aborted  <<< rviz_ogre_vendor [55min 10s]

Summary: 127 packages finished [56min 40s]
  1 package failed: rosidl_typesupport_fastrtps_cpp
  7 packages aborted: foonathan_memory_vendor pluginlib rcl_yaml_param_parser rmw_connext_shared_cpp rosidl_generator_cpp rosidl_typesupport_connext_cpp rviz_ogre_vendor
  5 packages had stderr output: foonathan_memory_vendor mimick_vendor rmw_connext_shared_cpp rosidl_typesupport_connext_cpp rosidl_typesupport_fastrtps_cpp
  173 packages not processed