from scapy.all import Ether, IP, ICMP, sendp, Dot1Q
from async_packet_test.context import make_pytest_context
from async_packet_test.predicates import saw_vlan_tag

context = make_pytest_context()

packet = Ether() / Dot1Q(vlan=102) / IP() / ICMP()


def test_saw_vlan_tag_veth2(context):
    result = context.expect('veth2', saw_vlan_tag(102))
    sendp(packet, iface='veth0')
    result.assert_true()


def test_did_not_see_vlan_anywhere_else(context):
    results = [
        context.expect('veth4', saw_vlan_tag(102)),
        context.expect('veth6', saw_vlan_tag(102)),
        context.expect('veth8', saw_vlan_tag(102)),
        context.expect('veth10', saw_vlan_tag(102)),
        context.expect('veth12', saw_vlan_tag(102)),
        context.expect('veth14', saw_vlan_tag(102))
    ]

    sendp(packet, iface='veth0')

    for result in results:
        assert not result
