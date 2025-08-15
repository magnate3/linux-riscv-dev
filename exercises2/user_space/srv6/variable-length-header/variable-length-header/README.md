# Introduction

It is not always obvious how best to handle packet headers with
variable length in P4 programs.  This document describes a couple of
possible approaches.


## Headers with variable-length fields with type `varbit`

P4_16 provides a type `varbit` that can be used within a packet header
definition, to indicate a field that has a length that can be
different from one packet to another.  The length of such a field is
not known before the packet is received and parsed.

Given the capabilities defined in the P4_16 language specification
version 1.1.0, the length of such a `varbit` field must be determined
before the header with the `varbit` field is extracted, typically from
the values of earlier fixed-length fields within the packet.

After such a header is extracted, the P4_16 language specification
does not require that an implementation allow you to _do_ much of
anything with such a `varbit` type field.  You can do these things:

+ Emit the header in a deparser, with the contents of the `varbit`
  field identical to what was received and parsed.
+ Delete the header containing the `varbit` field, e.g. by calling the
  `setInvalid` method on the header.
+ Copy the entire valid header to another header of the same type,
  perhaps emitting it as well as the header it was copied from.
+ Copy the `varbit` field to another variable of type `varbit`, if it
  has the same maximum width in bits.

Note that one could declare a variable with a header type that
contains a `varbit` field, and call the `setValid` method on that
variable.  The P4_16 language specification does not mandate the
contents nor the length of the `varbit` field(s) of such a header (see
Section 8.8 "Operations on variable-size bit types"), so emitting such
a header seems like it would lead to unpredictable output packets, or
at least implementation-specific behavior.

Other operations that the P4_16 language specification does not
mandate, nor even mention:

+ Use of a `varbit` field as a key of a table or parser value set.
+ Assignment of a literal value to a variable with type `varbit`.
+ Modifying the dynamic length of a `varbit` field, by any means other
  than assignment from another variable of type `varbit` which copies
  the current dynamic length _and_ the contents.
+ Arithmetic operations.
+ Extracting some of the bits.
+ Modifying some of the bits.

Basically, the operations required by the language specification let
you remove a header containing a field with type `varbit`, pass it
through to the output packet without modification, or make copies of
that header (and perhaps modify the contents of the fixed-length
fields, but not the `varbit` field).

You cannot look at `varbit` values in any meaningful way, nor change
their values.  There is also no way to add a new header to a packet
with a field of type `varbit` and initialize the value of a field with
a type `varbit` (except by copying the `varbit` value from another
variable of type `varbit`).

This makes them useful for a few things, such as:

+ Receive a packet with an IPv4 header that contains options, and
  preserve the contents of the IPv4 options in the outgoing packet.

This use case should not be dismissed -- it is an important use case
for processing IPv4 headers that may contain IPv4 options, and many
switches do forward such packets without examining the contents of
their options (or punt them to a nearby general purpose CPU for
significantly slower processing in software).

[Section 12.8.2 "Variable width extraction"](https://p4.org/p4-spec/docs/P4-16-v1.1.0-spec.html#sec-packet-extract-two)
of the P4_16 language specification shows one way to check the value
of the IHL field of an IPv4 header before using it to calculate the
length of the IPv4 options, and then extract them into a header
containing a `varbit` field.

Assuming that the restrictions above are accurate, if you want to
write a P4 program where:

+ it can receive and parse packets with IPv4 headers containing
  options that are stored in a `varbit` field at the end of the Ipv4
  header, and
+ it can _add_ an IPv4 header to a received packet, with all bits in
  the output header containing values that have predictable contents
  across P4 implementations

Then it seems that one must define two separate header types for an
IPv4 header:

+ one containing a `varbit` field, used for parsing received IPv4
  headers that may contain options.
+ one that contains no `varbit` fields, used for constructing new IPv4
  headers for the output packet that were not part of the received
  packet.


## Alternate approach: use multiple fixed-length headers

I will use as an example of this a P4 program where one wants to
examine the contents of, and possibly modify, IPv6 packets that
contain extension headers for IPv6 Segment Routing.  Such headers are
defined in an Internet Draft as of November 2018, where the latest
version of the draft is:

+ "IPv6 Segment Routing Header (SRH)", Internet-Draft, C. Filsfils,
  S. Previdi, J. Leddy, S. Matsushima, and D. Voyer, October 22, 2018,
  Expires: April 25, 2019,
  https://tools.ietf.org/html/draft-ietf-6man-segment-routing-header-15

One possible sequence of headers for an IPv6 packet with an IPv6
Segment Routing extension header, carried over an Ethernet link, is as
follows:

+ Ethernet with protocol field value of 0x86dd, indicating the next
  header is IPv6.
+ IPv6 with Next Header field value of 43 decimal, indicating the next
  header is an IPv6 Routing extension header.
+ IPv6 Segment Routing header, with a Next Header field value of 17
  decimal, indicating the next header is UDP.

A copy of the ASCII art diagram from the Internet draft showing the
contents of an IPv6 Segment Routing header is given below:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Next Header   |  Hdr Ext Len  | Routing Type  | Segments Left |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Last Entry   |     Flags     |              Tag              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|            Segment List[0] (128 bits IPv6 address)            |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                                                               |
                              ...
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|            Segment List[n] (128 bits IPv6 address)            |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//                                                             //
//         Optional Type Length Value objects (variable)       //
//                                                             //
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

The example P4 programs given here are _not_ intended to be anything
near to a complete implementation of a router that performs IPv6
segment routing.  It is intended to demonstrate one way to write a
P4_16 parser that can parse some simple cases of such a header, one
example of modifying the contents of such a header, and a deparser
that emits the header (either the original one, or containing whatever
modifications were made by the P4 code).

There are two variations of such a "skeleton" program:

+ The program [`srv6-skeleton.p4`](srv6-skeleton.p4) is the more
  general technique, where you create multiple different `header`
  types, one for each format of that header you want to support.
+ The program [`srv6-skeleton-v2.p4`](srv6-skeleton-v2.p4) takes
  advantage of the fact that for this example, the variable length
  header consists of a fixed length portion that always has the same
  format, follows by 1 to N instances of a portion of the header that
  all have the same format and size as each other, and thus can be
  represented via a P4_16 header stack.

See the file [`README-testing.md`](README-testing.md) for how to use
Scapy to construct packets containing IPv6 Routing extension headers,
which I used to test the skeleton P4 programs above using `p4c` and
`simple_switch`.

Code that has many similar cases for handling different fixed length
headers can get fairly repetitive.  See the article [Generating P4
code for fun and profit](/code-generation/README.md) for approaches
you can use to help in writing such repetitive code.
