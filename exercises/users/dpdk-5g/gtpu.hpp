#pragma once
/**
@file
GTPv1-U protocol definition in med (https://github.com/cppden/med)
3GPP TS 29.281 V17.1.0 (2021-09)
NOTE: The complete range of message types defined for GTPv1 is defined in 3GPP TS 29.060.

@copyright Denis Priyomov 2016-2022
Distributed under the MIT License
(See accompanying file LICENSE or visit https://github.com/cppden/med)
*/

#include <arpa/inet.h>

#include "med/value.hpp"
#include "med/octet_string.hpp"
#include "med/sequence.hpp"
#include "med/mandatory.hpp"
#include "med/optional.hpp"
#include "med/placeholder.hpp"
#include "med/choice.hpp"

namespace gtpu {

constexpr uint16_t UDP_PORT = 2152;

template <typename ...T>
using M = med::mandatory<T...>;
template <typename ...T>
using O = med::optional<T...>;
template <uint8_t TAG>
using T = med::value<med::fixed<TAG, uint8_t>>;
template <class T>
using L = med::length_t<T>;
template <class T>
using CASE = M<med::value<med::fixed<T::id, uint8_t>>, T>;

/*
5	GTP-U header
5.1	General format
Always present fields:
- Version field: This field is used to determine the version of the GTP-U protocol. The version number shall be
	set to '1'.
- Protocol Type (PT): This bit is used as a protocol discriminator between GTP (when PT is '1') and GTP' (when
	PT is '0').
- Extension Header flag (E): This flag indicates the presence of a meaningful value of the Next Extension Header
	field. When it is set to '0', the Next Extension Header field either is not present or, if present, shall not
	be interpreted. When it is set to '1', the Next Extension Header field is present, and shall be interpreted,
	as described below in this section.
- Sequence number flag (S): This flag indicates the presence of a meaningful value of the Sequence Number field.
	When it is set to '0', the Sequence Number field either is not present or, if present, shall not be interpreted.
	When it is set to '1', the Sequence Number field is present, and shall be interpreted, as described below.
	For the Echo Request, Echo Response, Error Indication and Supported Extension Headers Notification messages,
	the S flag shall be set to '1'. Since the use of Sequence Numbers is optional for G-PDUs, the PGW, SGW, ePDG,
	eNodeB and TWAN should set the flag to '0'. However, when a G-PDU (T-PDU+header) is being relayed by the
	Indirect Data Forwarding for Inter RAT HO procedure, then if the received G-PDU has the S flag set to '1', then
	the relaying entity shall set S flag to '1' and forward the G-PDU (T-PDU+header). In an End marker message the
	S flag shall be set to '0'.
- N-PDU Number flag (PN): This flag indicates the presence of a meaningful value of the N-PDU Number field. When it
	is set to '0', the N-PDU Number field either is not present, or, if present, shall not be interpreted. When it
	is set to '1', the N-PDU Number field is present, and shall be interpreted, as described below in this section.
- Message Type: This field indicates the type of GTP-U message.
- Length: This field indicates the length in octets of the payload, i.e. the rest of the packet following the
	mandatory part of the GTP header (that is the first 8 octets). The Sequence Number, the N-PDU Number or any
	Extension headers shall be considered to be part of the payload, i.e. included in the length count.
- Tunnel Endpoint Identifier (TEID): This field unambiguously identifies a tunnel endpoint in the receiving GTP‑U
	protocol entity. The receiving end side of a GTP tunnel locally assigns the TEID value the transmitting side
	has to use. The TEID shall be used by the receiving entity to find the PDP context, except for the following cases:
	*	The Echo Request/Response and Supported Extension Headers notification messages, where the Tunnel Endpoint
		Identifier shall be set to all zeroes.
	*	The Error Indication message where the Tunnel Endpoint Identifier shall be set to all zeros.
- When setting up a GTP-U tunnel, the GTP-U entity shall not assign the value 'all zeros' to its own TEID. However,
	for backward compatibility, if a GTP-U entity receives (via respective control plane message) a peer's TEID that
	is set to the value 'all zeros', the GTP-U entity shall accept this value as valid and send the subsequent G-PDU
	with the TEID field in the header set to the value 'all zeros'.

Optional fields:
- Sequence Number: If Sequence Number field is used for G-PDUs (T-PDUs+headers), an increasing sequence number for
	T-PDUs is transmitted via GTP-U tunnels, when transmission order must be preserved. For Supported Extension
	Headers Notification and Error Indication messages, the Sequence Number shall be ignored by the receiver, even
	though the S flag is set to '1'.
- N-PDU Number: This field is used at the Inter SGSN Routeing Area Update procedure and some inter-system handover
	procedures (e.g. between 2G and 3G radio access networks). This field is used to co-ordinate the data transmission
	for acknowledged mode of communication between the MS and the SGSN. The exact meaning of this field depends upon
	the scenario. (For example, for GSM/GPRS to GSM/GPRS, the SNDCP N-PDU number is present in this field).
- Next Extension Header Type: This field defines the type of Extension Header that follows this field in the GTP‑PDU.
*/
struct sequence_number : med::value<uint16_t>
{
	static constexpr char const* name() { return "Sequence Number"; }
};

struct npdu_number : med::value<uint8_t>
{
	static constexpr char const* name() { return "N-PDU Number"; }
};

namespace ext {

//Figure 5.2.1-3: Definition of Extension Header Type
struct header_type : med::value<uint8_t>
{
#ifdef CODEC_TRACE_ENABLE
	static constexpr char const* name() { return "Extension Header Type"; }
#endif
};

/*
5.2	GTP-U Extension Header
5.2.1	General format of the GTP-U Extension Header
- Extension Header Length: specifies the length of the particular Extension header in 4 octets units.
- Next Extension Header Type: specifies the type of any Extension Header that may follow a particular Extension Header.
	Bits 7 and 8 of the Next Extension Header Type define how the recipient shall handle unknown Extension Types.
	The recipient of an extension header of unknown type but marked as 'comprehension not required' for that recipient
	shall read the 'Next Extension Header Type' field (using the Extension Header Length field to identify its location
	in the GTP-PDU).

Oct
-----+----------------------------+
   1 | Extension Header Length    |
 2-m | Extension Header Content   |
 m+1 | Next Extension Header Type |

*/

/*
 * next extension header type comprehension bits:
 * Figure 5.2.1-2: Definition of bits 7 and 8 of the Extension Header Type
 */
//Comprehension of this extension header is required by the Endpoint Receiver but not by an Intermediate Node.
constexpr bool required(uint8_t eh_type)    { return 0 != (eh_type & 0b10000000); }

//true: An Intermediate Node shall forward it to any Receiver Endpoint.
//false: shall discard the Extension Header Content and not forward it to any Receiver Endpoint.
constexpr bool forward(uint8_t eh_type)     { return 0 == (eh_type & 0b01000000); }

struct no_more : med::empty<> {};

struct length : med::value<uint8_t, med::padding<uint32_t, med::offset<2>>> //in 4 octets
{
	static void value_to_length(std::size_t& v)
	{
		v = v * 4 - 2; //exclude tag + len bytes
	}

	static void length_to_value(std::size_t& v)
	{
		v = (v + 5) / 4; //include tag + len bytes
	}

#ifdef CODEC_TRACE_ENABLE
	static constexpr char const* name() { return "Extension Header Length"; }
#endif
};

/*
5.2.2.1	UDP Port
This extension header may be transmitted in Error Indication messages to provide the UDP Source Port of the G-PDU
that triggered the Error Indication. It is 4 octets long, and therefore the Length field has value 1.
Octet|
   1 | 0x01
 2-3 | UDP Port number
   4 | Next Extension Header Type
*/
struct udp_port : med::value<uint16_t>
{
    static constexpr char const* name() { return "UDP Port Number"; }
};

/*
5.2.2.2	PDCP PDU Number
This extension header is transmitted, for example in UTRAN, at SRNS relocation time, to provide the PDCP sequence
number of not yet acknowledged N-PDUs. It is 4 octets long, and therefore the Length field has value 1.
When used during a handover procedure between two eNBs at the X2 interface (direct DL data forwarding) or via the
S1 interface (indirect DL data forwarding) in E-UTRAN, bit 8 of octet 2 is spare and shall be set to zero.
NOTE 1: The PDCP PDU number field of the PDCP PDU number extension header has a maximum value which requires 15 bits
(see 3GPP TS 36.323 [24]); thus, bit 8 of octet 2 is spare.

Octet|
   1 | 0x01
 2-3 | PDCP PDU number
   4 | Next Extension Header Type
*/
struct pdcp_pdu : med::value<uint16_t>
{
    static constexpr char const* name() { return "PDCP PDU Number"; }
};

/*
5.2.2.2A	Long PDCP PDU Number
This extension header is used for direct X2 or indirect S1 DL data forwarding during a Handover procedure between
two eNBs. The Long PDCP PDU number extension header is 8 octets long, and therefore the Length field has value 2.
The PDCP PDU number field of the Long PDCP PDU number extension header has a maximum value which requires 18 bits
(see 3GPP TS 36.323 [24]). Bit 2 of octet 2 is the most significant bit and bit 1 of octet 4 is the least significant
bit, see Figure 5.2.2.2A-1. Bits 8 to 3 of octet 2, and Bits 8 to 1 of octets 5 to 7 shall be set to 0.
NOTE: A G-PDU which includes a PDCP PDU Number contains either the extension header PDCP PDU Number or Long PDCP
PDU Number

Oct\Bit | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
--------+---+---+---+---+---+---+---+---+
 1      |            0x02
 2      | Spare                 |PDCP PDU
        |                       | number
 3-4    | PDCP PDU number
 5-7    | Spare
 8      | Next Extension Header Type
*/
struct long_pdcp_pdu : med::value<med::bits<24>>
{
	value_type get() const              { return get_encoded() & 0x3FFFF; }
	void set(value_type v)              { set_encoded(v & 0x3FFFF); }
	static constexpr char const* name() { return "Long PDCP PDU Number"; }
};

/*
5.2.2.3	Service Class Indicator
This extension header identifies the service class indicator (SCI) associated with the T-PDU carried by the downlink
G-PDU. This information may be used by the A/Gb mode GERAN access for improved radio utilisation (see clause 5.3.5.3
of 3GPP TS 23.060 [4]).
In this version of the specification, this extension header may be transmitted over the Gn/Gp, S5/S8 and S4 interface.
An eNodeB, RNC or MME shall ignore this information if received over the S1-U, S12, Iu, S11-U or any other interfaces
not defined above, but still shall handle the G-PDU.
NOTE1:	This extension header is also sent over the S1-U, S12, Iu interface and S11-U if the SGW receives the Service
Class Indicator from S5/S8 for a UE having a user plane connection with an RNC or an eNodeB. This can happen when the
PGW does not have an accurate knowledge of the current RAT of the user e.g. after a handover from GERAN to (E)UTRAN.
It is 4 octets long and therefore the Length field has the value 1.

Octet|
  1  | 0x01
  2  | Service Class Indicator
  3  | Spare
  4  | Next Extension Header Type

If the bit 8 of octet 2 is set to 0, this indicates an operator specific Service Class Indicator value is included.
Otherwise, it shall indicate that a standardised SCI is included.
NOTE 2: No standardized SCI value is defined in this release, it is intended to standardize SCIs in a future release.
Bits 8 to 1 of the octet 2 represent the binary coded value of the SCI, applications with similar Radio Resource
Management treatment in GERAN shall be represented by the same value.
The octet 2 is coded as shown in Table 5.2.2.3-1.
Bits 8 to 1 of the octet 3 are spare bits and shall be set to zero.
    Table 5.2.2.3-1: Service Class Indicator (SCI, octet 2)
=======================================
Bit-8 = 0        Operator-specific SCI
---------------------------------------
Bits-7..1
0000000..0001111 Operator-specific SCIs
0010000..1111111 Spare for future use
=======================================
Bit-8 = 1        Standardised SCI
---------------------------------------
Bits-7..1
0000000..1111111 Spare for future use
=======================================
*/
struct sci : med::value<uint8_t>
{
	static constexpr char const* name() { return "Service Class Indicator"; }
	value_type get() const              { return get_encoded() & 0x7F; }
	bool specific() const               { return !(get_encoded() & 0x80); }
};

/*
5.2.2.4	RAN Container
This extension header may be transmitted in a G-PDU over the X2 user plane interface between the eNBs.
The RAN Container has a variable length and its content is specified in 3GPP TS 36.425 [25].
A G-PDU message with this extension header may be sent without a T-PDU.

 Octets |
    1   | 0xN
2-(4N-1)| RAN Container
   4N   | Next Extension Header Type
*/
struct container_data : med::octet_string<> {};

struct ran_container : container_data
{
	static constexpr char const* name() { return "RAN Container"; }
};

struct xw_ran_container : container_data
{
	static constexpr char const* name() { return "Xw RAN Container"; }
};

struct nr_ran_container : container_data
{
	static constexpr char const* name() { return "NR RAN Container"; }
};

struct pdu_session_container : container_data
{
	static constexpr char const* name() { return "PDU Session Container"; }
};

struct any_tag : med::value<uint8_t>//, med::optional_t
{
	static constexpr char const* name()     { return "Any"; }
	static constexpr bool match(value_type) { return true; }
};

struct unknown : med::sequence<
	M< any_tag >,
	M< container_data >
>
{};

struct header : med::choice<
	M< T<0b0000'0000>, no_more >,
	M< T<0b0100'0000>, L<ext::length>, udp_port >,
	M< T<0b1100'0000>, L<ext::length>, pdcp_pdu >,
	M< T<0b1000'0010>, L<ext::length>, long_pdcp_pdu >,
	M< T<0b0010'0000>, L<ext::length>, sci >,
	M< T<0b1000'0001>, L<ext::length>, ran_container >,
	M< T<0b1000'0011>, L<ext::length>, xw_ran_container >,
	M< T<0b1000'0100>, L<ext::length>, nr_ran_container >,
	M< T<0b1000'0101>, L<ext::length>, pdu_session_container >,
	M< any_tag, L<ext::length>, unknown >
>
{
#ifdef CODEC_TRACE_ENABLE
	static constexpr char const* name() { return "Extension Header"; }
#endif

	struct setter
	{
		template <class IEs>
		void operator()(npdu_number& out, IEs const& ies) const
		{
			if (!ies.template as<npdu_number>().is_set())
			{
				if (ies.template as<ext::header>().is_set()
				|| ies.template as<sequence_number>().is_set())
				{
					out.set(0);
				}
			}
		}

		template <class IEs>
		void operator()(sequence_number& out, IEs const& ies) const
		{
			if (!ies.template as<sequence_number>().is_set())
			{
				if (ies.template as<ext::header>().is_set()
				|| ies.template as<npdu_number>().is_set())
				{
					out.set(0);
				}
			}
		}

		template <class IEs>
		void operator()(header& out, IEs const& ies) const
		{
			if (!ies.template as<header>().is_set())
			{
				if (ies.template as<sequence_number>().is_set()
				|| ies.template as<npdu_number>().is_set())
				{
					out.template ref<no_more>();
				}
			}
		}
	};
};

struct next_header : header
{
	struct has_next
	{
		template <class IES>
		constexpr bool operator()(IES const& ies) const
		{
			//if no chain then check ext::header otherwise look into the last ext::next_header
			//if previous next_eh_type wasn't no_more then we need one more next_eh_type
			auto& me = ies.template as<next_header>();
#ifdef CODEC_TRACE_ENABLE
			if (me.empty())
			{
				auto* ptr = ies.template as<header>().template get<no_more>();
				CODEC_TRACE("no-more = %p", (void*)ptr);
			}
			else
			{
				auto* ptr = me.last()->template get<no_more>();
				CODEC_TRACE("no-more = %p", (void*)ptr);
			}
#endif
			bool const res = nullptr == (me.empty()
				? ies.template as<header>().template get<no_more>()
				: me.last()->template get<no_more>());
			CODEC_TRACE("ext::%sheader -> %d", me.empty() ? "":"next_", res);
			return res;
		}
	};

#ifdef CODEC_TRACE_ENABLE
	static constexpr char const* name() { return "Next Extension Header"; }
#endif
};

} //end: namespace ext

/*
Oct\Bit | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
--------+---+---+---+---+---+---+---+---+
    1   | Version   |PT | 0 | E | S |PN |
    2   | Message Type                  |
    3   | Length (1st Octet)            |
    4   | Length (2nd Octet)            |
    5   | TEID (1st Octet)              |
    6   | TEID (2nd Octet)              |
    7   | TEID (3rd Octet)              |
    8   | TEID (4th Octet)              |
    9   | SN (1st Octet) [NOTE 1,4]     |
   10   | SN (2nd Octet) [NOTE 1,4]     |
   11   | N-PDU Number [NOTE 2,4]       |
   12   | Next Ext Hdr Type [NOTE 3,4]  |

NOTE 1: This field shall only be evaluated when indicated by the S flag set to 1.
NOTE 2: This field shall only be evaluated when indicated by the PN flag set to 1.
NOTE 3: This field shall only be evaluated when indicated by the E flag set to 1.
NOTE 4: This field shall be present if and only if any one or more of the S, PN and E flags are set.
*/
struct opt_header : med::sequence<
	M< sequence_number, ext::header::setter >,
	M< npdu_number, ext::header::setter >,
	M< ext::header, ext::header::setter >,
	O< ext::next_header, ext::next_header::has_next, med::max<8> >
>
{
};

struct version_flags : med::value<uint8_t>
{
	enum : value_type
	{
		VER = 0b11100000, //Version
		PT  = 0b00010000, //Protocol Type: 1-GPTU, 0-GTP'
		E   = 0b00000100, //Extension Header flag
		S   = 0b00000010, //Sequence number flag
		NP  = 0b00000001, //N-PDU Number flag
	};

	static constexpr char const* name() { return "Version/Flags"; }

	template <std::size_t N>
	void print(char (&sz)[N]) const
	{
		value_type const flags = get();
#define PB(bit) (flags&bit)?(#bit)[0]:'.'
		std::snprintf(sz, sizeof(sz), "v%u[%c%c%c%c]", (flags&VER)>>5, PB(PT), PB(E), PB(S), PB(NP));
#undef PB
	}

	struct has_ext_fields
	{
		template <class HDR>
		bool operator()(HDR const& hdr) const
		{
			return hdr.template as<version_flags>().get() & (E|S|NP);
		}
	};

	struct setter
	{
		template <class IEs>
		void operator()(version_flags& out, IEs const& ies) const
		{
			auto& opt = ies.template as<opt_header>();
			auto& eh = opt.template get<ext::header>();
			auto& sn = opt.template get<sequence_number>();
			auto& np = opt.template get<npdu_number>();
			auto const vf =
				(1 << 5) | //version = 1
				PT | //protocol type = 1 (GTPU)
				(eh.is_set() ? E:0) |
				(sn.is_set() ? S:0) |
				(np.is_set() ? NP:0);
			out.set_encoded(vf);
		}

	};
};

struct message_type : med::value<uint8_t>
{
#ifdef CODEC_TRACE_ENABLE
	static constexpr char const* name() { return "Message Type"; }
#endif
};

struct teid : med::value<uint32_t>
{
	static constexpr char const* name() { return "TEID"; }
};

struct header : med::sequence<
	M< version_flags, version_flags::setter >,
	M< message_type >,
	med::placeholder::_length<8>,
	M< teid >,
	O< opt_header, version_flags::has_ext_fields >
>
{
	std::size_t get_tag() const             { return get<message_type>().get(); }
	void set_tag(std::size_t tag)           { ref<message_type>().set(tag); }

	uint8_t vf() const                      { return get<version_flags>().get(); }
	uint8_t version() const                 { return (vf() & version_flags::VER) >> 5; }
	bool is_gtpu() const                    { return 0 != (vf() & version_flags::PT); }

	teid::value_type get_teid() const       { return get<teid>().get(); }
	void set_teid(teid::value_type v)       { ref<teid>().set(v); }

	sequence_number::value_type sn() const  { return (vf() & version_flags::S) ? get<opt_header>()->get<sequence_number>().get() : 0; }
	void sn(sequence_number::value_type v)  { ref<opt_header>().ref<sequence_number>().set(v); }

	npdu_number::value_type npdu() const    { return (vf() & version_flags::NP) ? get<opt_header>()->get<npdu_number>().get() : 0; }
	void npdu(npdu_number::value_type v)    { ref<opt_header>().ref<npdu_number>().set(v); }

	static constexpr char const* name()     { return "Header"; }
};

/*
7.1.1	Presence requirements of Information Elements
There are three different presence requirements (Mandatory, Conditional, or Optional) for an IE within a given GTP-PDU:
- Mandatory means that the IE shall be included by the sending side, and that the receiver diagnoses a "Mandatory IE
	missing" error when detecting that the IE is not present.
- Conditional means:
	* that inclusion of the IE by the sender depends on conditions specified in the relevant protocol specification;
	* that the receiver can expect that the IE is present based on its parameter combination in the message and/or on
		the state of the receiving node.
- Optional means that the IE shall be included as a service option. Therefore, the IE may be included or not in a message.


The information elements shall be sorted, with the Type fields in ascending order, in the signalling messages.
The Length field contains the length of the information element excluding the Type and Length field.
The most significant bit in the Type field is set to 0 when the TV format is used and set to 1 for the TLV format.
*/

/*
8.2	Recovery
The value of the restart counter shall be set to 0 by the sending entity and ignored by the receiving entity.
This information element is used in GTP user plane due to backwards compatibility reasons.
*/
struct recovery : med::value<uint8_t>
{
	using tag = med::value<med::fixed<14, uint8_t>>;
	static constexpr char const* name() { return "Restart Counter"; }
};

/*
8.3	Tunnel Endpoint Identifier Data I
The TEID Data I information element contains the TEID used by a GTP entity for the user plane.
*/
struct teid_data : med::value<uint32_t>
{
	using tag = med::value<med::fixed<16, uint8_t>>;
	static constexpr char const* name() { return "TEID Data I"; }
};

/*
8.4	GTP-U Peer Address
The GTP-U peer Address information element contains the address of a GTP. The Length field may have only two values
(4 or 16) that determine if the Value field contains IPv4 or IPv6 address.
The IPv4 address structure is defined in RFC 791 [10].
The IPv6 address structure is defined in RFC 4291 [11].
The encoded address might belong not only to a GSN, but also to an RNC, eNodeB, SGW, ePDG, PGW or TWAN.
*/
struct peer_address : med::octet_string<med::octets_var_intern<16>, med::min<4>>
{
	using tag = med::value<med::fixed<133, uint8_t>>;
	using length = med::value<uint16_t>;

	static constexpr char const* name()         { return "Peer Address"; }
	template <std::size_t N>
	void print(char (&sz)[N]) const
	{
		uint8_t const* p = this->data();
		if (4 == this->size())
		{
			std::snprintf(sz, sizeof(sz), "%u.%u.%u.%u", p[0], p[1], p[2], p[3]);
		}
		else
		{
			struct in6_addr in_addr;
			std::memcpy(in_addr.s6_addr, p, sizeof(in_addr.s6_addr));
			inet_ntop(AF_INET6, in_addr.s6_addr, sz, sizeof(sz)-1);
		}
	}
};

/*
8.5	Extension Header Type List
This information element contains a list of 'n' Extension Header Types. The length field is set to the number of
extension header types included.
*/
struct eh_type_list : med::sequence<
	M< ext::header_type, med::max<8> >
>
{
	using tag = med::value<med::fixed<141, uint8_t>>;
	using length = med::value<uint8_t>;
	static constexpr char const* name() { return "Extension Header Type List"; }
};

/*
8.6	Private Extension
The Private Extension information element contains vendor specific information. The Extension Identifier is a value
defined in the Private Enterprise number list in the most recent "Assigned Numbers" RFC (RFC 1700 or later).
This is an optional information element that may be included in any GTP Signalling message. A signalling message may
include more than one information element of the Private Extension type.
*/
struct extension_id : med::value<uint16_t>
{
	static constexpr char const* name() { return "Extension Identifier"; }
};

struct extension_value : med::octet_string<>
{
	static constexpr char const* name() { return "Extension Value"; }
};

struct private_extension : med::sequence<
	M< extension_id >,
	M< extension_value >
>
{
	using tag = med::value<med::fixed<255, uint8_t>>;
	using length = med::value<uint16_t>;
	static constexpr char const* name() { return "Private Extension"; }
};


/* 7.2	Path Management Messages */

/*
7.2.1	Echo Request
A GTP-U peer may send an Echo Request on a path to the other GTP-U peer to find out if it is alive (see section Path
Failure). Echo Request messages may be sent for each path in use. A path is considered to be in use if at least one
PDP context, EPS Bearer, MBMS UE context, or MBMS bearer context uses the path to the other GTP-U peer. When and how
often an Echo Request message may be sent is implementation specific but an Echo Request shall not be sent more often
than every 60 s on each path. This doesn’t prevent resending an Echo Request with the same sequence number according
to the T3-RESPONSE timer.
Even if there is no path in use, a  GTP-U peer shall be prepared to receive an Echo Request at any time and it shall
reply with an Echo Response.  The optional Private Extension contains vendor or operator specific information.
For the GTP-U tunnel setup between two nodes for forwarding user traffic, e.g. between eNodeBs for direct forwarding
over X2, Echo Request path maintenance message shall not be sent except if the forwarded data and the normal data are
sent over the same path.
*/
struct echo_request : med::sequence<
	O< private_extension::tag, L<private_extension::length>, private_extension, med::inf >
>
{
	static constexpr uint8_t id = 1;
	static constexpr char const* name() { return "Echo Request"; }
};

/*
7.2.2	Echo Response
The message shall be sent as a response to a received Echo Request.
The Restart Counter value in the Recovery information element shall not be used, i.e. it shall be set to zero by the
sender and shall be ignored by the receiver. The Recovery information element is mandatory due to backwards
compatibility reasons.
The optional Private Extension contains vendor or operator specific information.
*/
struct echo_response : med::sequence<
	M< recovery::tag, recovery >,
	O< private_extension::tag, L<private_extension::length>, private_extension, med::inf >
>
{
	static constexpr uint8_t id = 2;
	static constexpr char const* name() { return "Echo Response"; }
};

/*
7.2.3	Supported Extension Headers Notification
This message indicates a list of supported Extension Headers that the GTP entity on the identified IP address can
support. This message is sent only in case a GTP entity was required to interpret a mandatory Extension Header but
the GTP entity was not yet upgraded to support that extension header. The GTP endpoint sending this message is marked
as not enabled to support some extension headers (as derived from the supported extension header list). The peer GTP
entity may retry to use all the extension headers with that node, in an attempt to verify it has been upgraded.
Implementers should avoid repeated attempts to use unknown extension headers with an endpoint that has signalled its
inability to interpret them.
*/
struct supported_eh : med::sequence<
	M< eh_type_list::tag, L<eh_type_list::length>, eh_type_list >
>
{
	static constexpr uint8_t id = 31;
	static constexpr char const* name() { return "Supported Extension Headers Notification"; }
};


/* 7.3	Tunnel Management Messages */
/*
7.3.1	Error Indication
When a GTP-U node receives a G-PDU for which no EPS Bearer context, PDP context, MBMS Bearer context, or RAB exists,
the GTP-U node shall discard the G-PDU. If the TEID of the incoming G-PDU is different from the value 'all zeros' the
GTP-U node shall also return a GTP error indication to the originating node. GTP entities may include the "UDP Port"
extension header (Type 0x40), in order to simplify the implementation of mechanisms that can mitigate the risk of
Denial-of-Service attacks in some scenarios.
Handling of the received Error Indication is specified in 3GPP TS 23.007 [3].
The information element Tunnel Endpoint Identifier Data I shall be the TEID fetched from the G-PDU that triggered
this procedure.

The information element GTP-U Peer Address shall be the destination address (e.g. destination IP address, MBMS Bearer
Context) fetched from the original user data message that triggered this procedure. A GTP-U Peer Address can be a GGSN,
SGSN, RNC, PGW, SGW, ePDG, eNodeB, TWAN or MME address. The TEID and GTP-U peer Address together uniquely identify the
related PDP context, RAB or EPS bearer in the receiving node.
*/
struct error_indication : med::sequence<
	M< teid_data::tag, teid_data >,
	M< peer_address::tag, L<peer_address::length>, peer_address >,
	O< private_extension::tag, L<private_extension::length>, private_extension, med::inf >
>
{
	static constexpr uint8_t id = 26;
	static constexpr char const* name() { return "Error Indication"; }
};

/*
7.3.2	End Marker
The End Marker message(s) shall be sent after sending the last G-PDU that needs to be sent on a GTP-U tunnel as
specified in 3GPP TS 23.401 [5] or after receiving an End Marker Indication as specified in subclause 5.7.1 of 3GPP
TS 23.402 [23]. The End Marker message(s) shall be sent for each GTP-U tunnel, except for the case of an E-UTRAN
Initiated E-RAB modification procedure. During an E-UTRAN Initiated E-RAB modification procedure, the SGW shall send
End marker message(s) to the eNodeB on the old S1-U tunnel for the tunnel(s) that are switched, i.e. if the S1 eNodeB
F-TEID of the GTP-U tunnel provided by the MME in a Modify Bearer Request or Modify Access Bearer Request is not the
same as the one previously stored in the SGW. Each GTP-U tunnel is identified with a respective TEID value in the
GTP-U header. The End Marker message indicates the end of the payload stream on a given tunnel, i.e. a G-PDU that
arrives after an End Marker message on this tunnel may be silently discarded. Table 7.3.2-1 specifies the information
element included in the End Marker message.
If an End Marker message is received with a TEID for which there is no context, then the receiver shall ignore this
message.
An MME may receive End Marker packets over an S11-U tunnel during the following procedures:
- Inter-MME TAU procedure;
- Establishment of S1-U bearer during Data Transport in Control Plane CIoT EPS optimisation.
The MME shall discard the End Marker packets. The MME may also initiate the release of the corresponding S11-U
resources; however the release of the S11-U resources is implementation dependent.
*/
struct end_marker : med::sequence<
	O< private_extension::tag, L<private_extension::length>, private_extension, med::inf >
>
{
	static constexpr uint8_t id = 254;
	static constexpr char const* name() { return "End Marker"; }
};

struct g_pdu : med::octet_string<>
{
	static constexpr uint8_t id = 255;
	static constexpr char const* name() { return "G-PDU"; }
};

struct proto : med::choice< header
	, CASE< echo_request >
	, CASE< echo_response >
	, CASE< error_indication >
	, CASE< supported_eh >
	, CASE< end_marker >
	, CASE< g_pdu >
>
{
	using length_type = med::value<uint16_t>;
#ifdef CODEC_TRACE_ENABLE
	static constexpr char const* name() { return "GTPU"; }
#endif
};


//peek-preview w/o decoding (to speed-up processing of G-PDU)
struct header_s
{
	enum : uint8_t {
		OS = 0, //offset for SN
		ON = 2, //offset for NP
		OE = 3, //offset for E
		LH = OE+1, //additional size for long header
	};

	//TODO: replace with std::span (C++20)
	struct payload_s
	{
		union{
			uint8_t const*  rw{}; //read/write
			uint8_t*        ro; //read-only
		};
		size_t const    size{};

		explicit constexpr operator bool() const noexcept   { return !empty(); }
		constexpr bool empty() const noexcept               { return 0 == size; }
	};

	uint8_t  flags;
	uint8_t  message_type;
	uint16_t m_length;
	uint32_t m_teid;

	static header_s* from(void* p)             { return static_cast<header_s*>(p); }
	static header_s const* from(void const* p) { return static_cast<header_s const*>(p); }

	uint8_t get_version() const { return (flags & version_flags::VER) >> 5; }
	bool is_gtpu() const        { return flags & version_flags::PT; }
	bool has_eh() const         { return flags & version_flags::E; }
	bool has_sn() const         { return flags & version_flags::S; }
	bool has_np() const         { return flags & version_flags::NP; }
	bool is_long() const        { return flags & (version_flags::E|version_flags::S|version_flags::NP); }

	uint16_t length() const     { return ntohs(m_length);}
	void length(uint16_t len)   { m_length = htons(len); }
	uint32_t teid() const       { return ntohl(m_teid); }
	void teid(uint32_t teid)    { m_teid = htonl(teid); }
	uint16_t sn() const         { return has_sn() ? ((*beyond(OS) << 8) | *beyond(OS+1)) : 0; }
	uint8_t npdu() const        { return has_np() ? *beyond(ON) : 0; }
	uint8_t next_eh() const     { return has_eh() ? *beyond(OE) : 0; }

	bool is_gpdu() const        { return is_gtpu() && g_pdu::id == message_type; }

	//make G-PDU returning the pointer to its payload
	uint8_t* gpdu(uint32_t teid_, std::size_t payload_size)
	{
		flags = 0x30;
		message_type = g_pdu::id;
		length(payload_size);
		teid(teid_);
		return beyond(0);
	}
	uint8_t* gpdu(uint32_t teid_, uint16_t _sn, std::size_t payload_size)
	{
		flags = 0x30 | version_flags::S;
		message_type = g_pdu::id;
		length(payload_size + LH) ;
		teid(teid_);
		*beyond(OS) = uint8_t(_sn >> 8);
		*beyond(OS+1) = uint8_t(_sn);
		*beyond(ON) = 0;
		*beyond(OE) = 0;
		return beyond(LH);
	}
	uint8_t* gpdu(uint32_t teid_, uint16_t _sn, uint8_t _np, std::size_t payload_size)
	{
		flags = 0x30 | version_flags::S | version_flags::NP;
		message_type = g_pdu::id;
		length(payload_size + LH);
		teid(teid_);
		*beyond(OS)   = uint8_t(_sn >> 8);
		*beyond(OS+1) = uint8_t(_sn);
		*beyond(ON)   = _np;
		*beyond(OE)   = 0;
		return beyond(LH);
	}

	//returns G-PDU payload
	payload_s gpdu() const
	{
		//return (is_gpdu() && 0 == next_eh()) ? beyond(is_long() ? LH:0) : nullptr;
		if (is_gpdu())
		{
			if (0 == next_eh())
			{
				return {{beyond(is_long() ? LH : 0)},length()};
			}

			size_t payLoadOffset = OE+1; //we've checked EH, look at next byte (length)
			if ( 0 == *beyond(payLoadOffset)) // check whether next ext len is not 0
			{
				return {};
			}

			do
			{
				payLoadOffset += (*beyond(payLoadOffset)) << 2;// skip header size itself * 4;// skip header container
			} while ( *beyond(payLoadOffset-1));

			return {{beyond(payLoadOffset)}, length() - payLoadOffset};
		}
		return {};
	}

private:
	//access bytes beyond this
	uint8_t* beyond(std::size_t n)              { return reinterpret_cast<uint8_t*>(this) + sizeof(*this) + n; }
	uint8_t const* beyond(std::size_t n) const  { return const_cast<header_s*>(this)->beyond(n); }

} __attribute__((packed));

} //end: namespace gtpu
