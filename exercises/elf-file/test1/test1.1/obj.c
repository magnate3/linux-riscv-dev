/* compile, but don't link this file:
 * $ gcc -c obj.c
 */
int add5(int num)
{
	    return num + 5;
}
 
int add10(int num)
{
    num = add5(num);
    return add5(num);
}
