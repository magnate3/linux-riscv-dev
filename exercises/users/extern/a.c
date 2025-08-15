#include <stdio.h> 
#define CONFIG_PGTABLE_LEVELS 3
static const char * const PT_LEVEL_NAME[] = {
	"   ", "pgd", "p4d",
	CONFIG_PGTABLE_LEVELS > 3 ? "pud" : "pgd",
	CONFIG_PGTABLE_LEVELS > 2 ? "pmd" : "pgd",
	"pte"
};
int i = 3; 
int p(void) { 
    printf("%d\n",i);     
    return 0;      
 }
