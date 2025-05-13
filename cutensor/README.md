# cutensor

Downloads the cutensor tar from the appropriate url and unpacks it and provides the linking mechanism for projects that depend on it. If the tar is already in the downloads directory then it reuses it from there. This allows the build directory to be deleted without constantly redownloading.

## TODO

Make a cmake variable to set the location if the user already has a copy.