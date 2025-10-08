#include <stdio.h>
#include <stdlib.h>

#ifndef PERCEPTRON_
#define PERCEPTRON_
#define MAXSIZE 256

char * getHeader(char path[MAXSIZE]) {
    FILE *file = fopen(path, "r");
    char *str = malloc(sizeof(*file));
    fgets(str, MAXSIZE, file);
    fclose(file);
    return str;
}

struct cell {
    char* content[MAXSIZE];
    struct cell *next;
};


int main(int argc, char * argv[]){
    char *result = getHeader(argv[1]);
    puts(result);
    free(result);
    return EXIT_SUCCESS;
}

#endif