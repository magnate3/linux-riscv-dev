#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
	int num;
	
        int i;
       
	FILE *fptr;
	
	fptr = fopen("test1.txt","w");
	
	for(i = 0 ; i<1000000; i++){

		fprintf(fptr, "test\n");
		
	}
	fclose(fptr);
	
	fptr = fopen("test2.txt","w");
	
	for(i = 0 ; i<1000000; i++){
		fprintf(fptr, "test\n");

	}
	fclose(fptr);
	
	fptr = fopen("test3.txt","w");
	
	for( i = 0 ; i<1000000; i++){
		fprintf(fptr,"test\n");

	}
	fclose(fptr);
	
	fptr = fopen("test4.txt","w");
	
	for(i = 0 ; i<1000000; i++){
		fprintf(fptr, "test\n");

	}
	fclose(fptr);
	sleep(1);
	fptr = fopen("test5.txt","w");
	for(i = 0 ; i<1000000; i++){
		fprintf(fptr,"test\n");

	}
	fclose(fptr);
	
	fptr = fopen("test1.txt","r");

	char a[100];
	for(i = 0 ; i<1000000; i++){
		
		fgets(a, 100,fptr);
		printf("%s\n",a);
	}
	fclose(fptr);
	sleep(1);
	fptr = fopen("test2.txt","r");
	char b[100];
	for(i = 0 ; i<1000000; i++){
		
		fgets(b, 100,fptr);
		printf("%s\n",b);
	}
	fclose(fptr);
	sleep(1);
	fptr = fopen("test3.txt","r");
	char c[100];
	for(i = 0 ; i<1000000; i++){
		
		fgets(c, 100,fptr);
		printf("%s\n",c);
	}
	fclose(fptr);
	sleep(1);
	fptr = fopen("test4.txt","r");
	char d[100];
	for(i = 0 ; i<1000000; i++){
		
		fgets(d, 100,fptr);
		printf("%s\n",d);
	}
	fclose(fptr);
	sleep(1);
	fptr = fopen("test5.txt","r");
	char e[100];
	for(i = 0 ; i<1000000; i++){
		
		fgets(e, 100,fptr);
		printf("%s\n",e);
	}
	
	
	fclose(fptr);
	
}
