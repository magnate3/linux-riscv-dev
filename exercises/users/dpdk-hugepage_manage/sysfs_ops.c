#include "sysfs_ops.h"
#include <stdio.h>
#include <stdlib.h>

//parse sysfs value containing one integer value
int parse_sysfs_value(const char *filename, unsigned long *tmp)
{
	FILE *f;
	char buf[BUFSIZ];
	char *end = NULL;

	f = fopen(filename,"r");
	if(f==NULL){
		fprintf(stderr,"cannot not open \"%s\" to parse...\n",filename);
		return -1;
	}
	
	if(fgets(buf, sizeof(buf), f) == NULL){
		fprintf(stderr,"cannot not read \"%s\" to parse...\n",filename);
		goto OUT;
	}

	*tmp = strtoul(buf, &end, 0);
	if(buf[0] == '\0' || end == NULL || (*end != '\n')){
		fprintf(stderr,"cannot parse sysfs value \"%s\"...\n",filename);
		goto OUT; 
	}
	return 0;
	
	OUT:
		fclose(f);
		return -1;
}
