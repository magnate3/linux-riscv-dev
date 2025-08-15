
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#if defined(__APPLE__) && defined(__clang__)
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "parson.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *g_tests_path = "./";
const char* get_file_path(const char *filename);
static char * read_file(const char * file_path) {
    FILE *fp = NULL;
    size_t size_to_read = 0;
    size_t size_read = 0;
    long pos;
    char *file_contents;
    fp = fopen(file_path, "r");
    if (!fp) {
        assert(0);
        return NULL;
    }
    fseek(fp, 0L, SEEK_END);
    pos = ftell(fp);
    if (pos < 0) {
        fclose(fp);
        assert(0);
        return NULL;
    }
    size_to_read = pos;
    rewind(fp);
    file_contents = (char*)malloc(sizeof(char) * (size_to_read + 1));
    if (!file_contents) {
        fclose(fp);
        assert(0);
        return NULL;
    }
    size_read = fread(file_contents, 1, size_to_read, fp);
    if (size_read == 0 || ferror(fp)) {
        fclose(fp);
        free(file_contents);
        assert(0);
        return NULL;
    }
    fclose(fp);
    file_contents[size_read] = '\0';
    return file_contents;
}

const char* get_file_path(const char *filename) {
    static char path_buf[2048] = { 0 };
    memset(path_buf, 0, sizeof(path_buf));
    sprintf(path_buf, "%s/%s", g_tests_path, filename);
    return path_buf;
}

int main()
{
    JSON_Array *array;
    size_t i;
    const char * key ;
    JSON_Object *commit;
    JSON_Value * root_value = json_parse_file(get_file_path("network.json"));
    JSON_Object * root_object = json_object(root_value);
    JSON_Value * array_value;
    if(!root_value)
    {
         printf("root value is null \n");
         return 0;
    }
    printf("dot  %.128s,   ipv6:  %.128s,  mac : %.128s, ipv4 :  %.128s \n",
               json_object_get_string(root_object, "dot"),
               json_object_dotget_string(root_object, "ipv6"),
               json_object_get_string(root_object, "mac"),
               json_object_get_string(root_object, "ipv4"));
    
    printf("use dotget, dot:  %.128s,   ipv6:  %.128s,  mac : %.128s, ipv4 :  %.128s \n",
               json_object_dotget_string(root_object, "dot"),
               json_object_dotget_string(root_object, "ipv6"),
               json_object_dotget_string(root_object, "mac"),
               json_object_dotget_string(root_object, "ipv4"));
    /* getting array from root value and printing commit info */
    array = json_object_get_array(root_object,"interests");
    //array = json_object_get_array(root_object,"neigh");
    //array_value = json_object_get_value(root_value, "neigh");
    //TEST(json_array_get_wrapping_value(array) == array_value);
    printf("count %lu \n", json_array_get_count(array));
#if 1
    for (i = 0; i < json_array_get_count(array); i++) {
        printf("%s \n",  json_array_get_string(array, i));
        //commit = json_array_get_object(array, i);
        //printf("%s \n", json_object_get_string(commit, "172.17.0.2"));
        //if(json_string(commit))
        //     printf("%s \n",  json_string(commit));
     }
#endif
    commit = json_object_get_object(root_object,"neigh");
    //commit = json_object_get_object(root_object,"favorites");
    printf("neigh count %lu \n", json_object_get_count(commit));
    for (i = 0; i < json_object_get_count(commit); i++) {
        key = json_object_get_name(commit,i);
        printf("key %s, value %s \n",  key, json_object_get_string(commit,key));
    }
    json_value_free(root_value);
    return 0;
}
