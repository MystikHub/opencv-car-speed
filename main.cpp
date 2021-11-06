#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    // Make sure a single argument has been given
    if(argc != 2) {
        printf("usage: SpoonInspection.out <path-to-image-directory>\n");
        return -1;
    }

    return 0;
}