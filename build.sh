cd /code
rm -rf build_docker
mkdir -p build_docker
cd build_docker
cmake .. -DCMAKE_BUILD_TYPE=Release
