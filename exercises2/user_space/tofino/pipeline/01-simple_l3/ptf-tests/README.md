# PTF Tests for simple_l3.p4

This directory contains example PTF tests for the program simple_l3.p4

Even though the program is quite simple, these tests are nowhere close to be
complete -- there are many execution paths that are not exercised by these 
tests. One hope is that you might be able to identify some of these missing 
tests and add more as an exercise.

What we are attempting to demonstrate is one of the posssible ways how you
can structure and organize your tests. This is important, since even for a
simple program like this one, it is possisble to write dozens of not hundreds 
of tests, meaning that it does not make sense to keep them all in one file.

In organizing these tests we rely on the Python construct 

``` python
from <module> import *
```

that allows us to have "foundational" modules doing most of the "heavy lifting"
and smaller modules with the test cases.

The main "foundational" module here is the file `simple_l3.py`, which is based
on `~/tools/testbase.py` file. It imports all the necessary modules and  
contains the class `P4ProgramTest` that defines the `setUp()`, `tearDown()` and 
`cleanUp()` methods. The latter two methods are fairly generic, but the 
`setUp()` method is customized: it has some code that does the following:
   1. It creates convenient "shortcuts" for the table objects, in this case 
      those are `self.ipv4_host` and `self.ipv4_lpm` to access. This is optional
      and might or might not be appropriate for bigger programs, especially
      those that might not have unique table names
   2. It adds proper annotations to the table key/action fields so that their
      values can be specified in a convenient notation instead of generic
      hex humbers or byte strings. In the future, this might be solved at the 
      compiler level by defining proper annotations.
   3. It creates a list of tables that should be cleaned up before and after
      the test (in the `cleanup()` method). In the future this will go away
      once proper generic cleanup method is implemented
The P4ProgramTest class also contains a convenient method `program_table()` and
other utility methods might be added to it in the future.
      
The individual test modules, such as `basic_test.py` import all the symbols
from this foundational module, thus avoiding the duplication of the boilerplate
code. These modules contain one or more classes representing individual tests. 
Each of these classes is subclassed from `P4ProgramTest` and should define at
least the method `runTest()`. It is also possible to override/extend the 
`setUp()` and `tearDown()` methods, but that's rarely needed.

Note, how the tests in `basic_test.py` are parameterized, which makes it easy
to reuse the code to run the tests for the different IP addresses, etc.

The other set of tests inside the module `sample_setup.py` are somewhat 
different -- they all reeuse the same basic table programming, but then each
tests check the behavior of the system with reegards to a particular packet. 

To share the same setup, the module defines a new "setup" class (`TestGroup1`)
that exteends the `setUp()` method of the basic class witth the code that 
programs the desired configuration in the tables. The actual test classes 
are then subclassed from `TestGroup1` and are all very short -- they only 
define the test packet, the expected result and then check for it.

It is possible to further modularize the tests, even place indivitual tests 
into separate files. It is also possible to have a directory hierarchy -- ptf 
will scan the directories recursively trying to find any Python files that
deefine classes that have `runTest()` method.
