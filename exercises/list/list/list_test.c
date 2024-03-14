#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/slab.h>
#include <linux/string.h>
struct birthday 
{int day;	
 int month;	
 int year;               /*birthday structure*/
 char *gender;
 char *name; 	
 struct list_head list;	
	
	
};

/**
 * The following defines and initializes a list_head object named birthday_list
 ***/
static LIST_HEAD(birthday_list);
//#define LIST_HEAD_INIT(name) { &(name), &(name) }
//
/*
#define LIST_HEAD(name) \
  struct list_head name = LIST_HEAD_INIT(name)
*/
int simple_init(void)
{
	 
struct birthday *person_zero;
struct birthday *person_one;
struct birthday *person_two;
struct birthday *person_three;
struct birthday *ptr1;
printk(KERN_INFO "Loading Module\n");
 person_zero = kmalloc(sizeof(*person_zero),GFP_KERNEL);
 person_zero->day =2;	
 person_zero->month=10;	
 person_zero->year=1988;
 person_zero->gender="male";	
 person_zero->name="Tom";
 INIT_LIST_HEAD(&person_zero->list);

list_add_tail(&person_zero->list, &birthday_list);	

 person_one = kmalloc(sizeof(*person_one),GFP_KERNEL);
 person_one->day =5;	
 person_one->month=6;	
 person_one->year=1999;
 person_one->gender="female";	
 person_one->name="kim";
 
list_add_tail(&person_one->list, &birthday_list);

 person_two = kmalloc(sizeof(*person_two),GFP_KERNEL);
 person_two->day =20;	
 person_two->month=2;	
 person_two->year=1995;
 person_two->gender="male";	
 person_two->name="Adam";
 

list_add_tail(&person_two->list, &birthday_list);

 person_three = kmalloc(sizeof(*person_three),GFP_KERNEL);
 person_three->day =12;	
 person_three->month=1;	
 person_three->year=1971;
 person_three->gender="female";	
 person_three->name="Laura";
 

list_add_tail(&person_three->list, &birthday_list);



		
/* This code block is just to print out the members of the birthday_list*/
 list_for_each_entry(ptr1,&birthday_list,list){

  printk(KERN_INFO "Day: %d Month: %d Year: %d Gender: %s Name: %s \n",ptr1->day, ptr1->month, ptr1->year,ptr1->gender,ptr1->name );

     }

 return 0;
}

void simple_exit(void) {

/*This code block is for removing the person*/


struct birthday *ptr1,*next;
struct birthday *ptr;
printk(KERN_INFO "Removing Module\n");
list_for_each_entry_safe(ptr1,next,&birthday_list,list){
      list_del(&ptr1->list);
      printk(KERN_INFO "delete node Day: %d Month: %d Year: %d Gender: %s Name: %s \n",ptr1->day, ptr1->month, ptr1->year,ptr1->gender,ptr1->name );
      kfree(ptr1);
}



/* This is the updated birthday_list*/
 list_for_each_entry(ptr,&birthday_list,list){

  printk(KERN_INFO "Day: %d Month: %d Year: %d Gender: %s Name: %s \n",ptr->day, ptr->month, ptr->year,ptr->gender,ptr->name );

     }	
	

}

module_init( simple_init );
module_exit( simple_exit );

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel Data Structures");
MODULE_AUTHOR("SGG");
