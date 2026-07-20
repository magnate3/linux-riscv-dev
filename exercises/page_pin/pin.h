#ifndef __PIN_H__
#define __PIN_H__
struct test_pin_address {
	        unsigned long addr;
		        unsigned long size;
};

#define TEST_PIN                _IOW('W', 0, struct test_pin_address)
#define TEST_UNPIN              _IOW('W', 1, struct test_pin_address)
#define MAX_PIN_PAGE            100
#endif
