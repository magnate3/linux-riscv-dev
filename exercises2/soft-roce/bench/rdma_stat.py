from pyroute2.netlink import nlmsg, nla, NLA_F_NESTED
from pyroute2.netlink.nlsocket import AsyncNetlinkSocket, NetlinkRequest
from typing import Optional
import asyncio

# RDMA Netlink 协议常量, copy from iproute2
NLM_F_REQUEST = 0x1
NLM_F_ROOT = 0x100
NLM_F_MATCH = 0x200
NLM_F_DUMP = NLM_F_ROOT | NLM_F_MATCH
NETLINK_RDMA = 20
RDMA_NL_NLDEV = 5
RDMA_NLDEV_ATTR_DEV_INDEX = 1
RDMA_NLDEV_ATTR_DEV_NAME = 2
RDMA_NLDEV_ATTR_PORT_INDEX = 3
RDMA_NLDEV_ATTR_STAT_HWCOUNTERS = 80
RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY = 81
RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME = 82
RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_VALUE = 83
RDMA_NLDEV_CMD_STAT_GET = 17
NLM_F_ACK = 0x4

# #define RDMA_NL_GET_TYPE(client, op) ((client << 10) + op)
def RDMA_NL_GET_TYPE(client, op):
  return (client << 10) + op

RDMA_STATE_GET_TYPE = RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_STAT_GET)

class RdmaStatGetMsg(nlmsg):
  nla_map = ((RDMA_NLDEV_ATTR_DEV_INDEX, 'DevIndex', 'uint32'),
             (RDMA_NLDEV_ATTR_PORT_INDEX, 'PortIndex', 'uint32'))

  @staticmethod
  def new(devidx, portidx):
    msg = RdmaStatGetMsg()
    msg['attrs'] = [
      # 对应 nla attr (RDMA_NLDEV_ATTR_DEV_INDEX, devidx)
      ['DevIndex', devidx],
      ['PortIndex', portidx]]
    msg['header']['type'] = RDMA_STATE_GET_TYPE
    msg['header']['flags'] = NLM_F_REQUEST | NLM_F_ACK
    return msg

# iproute2 rdma stat 有 bug, 其使用了 u32
# strace -e 捕捉到的内核返回:
# \x68\x77\x5f\x72\x78\x5f\x62\x79\x74\x65\x73\x5f\x63\x6e\x74\x00\x0c\x00\x53\x00\xa2\x10\xbc\x24\x82\x02\x00\x00
# \x68\x77\x5f\x72\x78\x5f\x62\x79\x74\x65\x73\x5f\x63\x6e\x74, 即 hw_rx_bytes_cnt
# 其对应值为 \xa2\x10\xbc\x24\x82\x02\x00\x00. 但 rdma stat 输出:
# hw_rx_bytes_cnt 616304802, 对应着 \xa2\x10\xbc\x24.
# 搞得我排查了半天是不是我逻辑问题.

class RdmaStatGetResp(nlmsg):
  # 在 decode 时会将不在 nla_map 的 nla 映射为 UNKNOWN NLA
  # 对应: ('UNKNOWN', {'header': {'length': 12, 'type': 2}})
  nla_map = ((RDMA_NLDEV_ATTR_DEV_INDEX, 'DevIndex', 'uint32'),
             (RDMA_NLDEV_ATTR_DEV_NAME, 'DevName', 'asciiz'),
             (RDMA_NLDEV_ATTR_STAT_HWCOUNTERS, 'HWC', 'HWC'),
             (RDMA_NLDEV_ATTR_PORT_INDEX, 'PortIndex', 'uint32'))

  class HWC(nla):
    nla_flags = NLA_F_NESTED
    # 这里缺少 ',' 会导致异常. 但 pyroute2 decode 并不会输出任何信息,
    # 只是输出结果缺少 HWC attr.
    nla_map = ((RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY, 'HWCE', 'HWCE'),)

    # RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY
    # print 输出 ('HWCE', {'attrs': [('Name', 'listen_create_cnt'), ('Value', 0)]}, 32768)
    # 见 nla_slot.__repr__, 32768 是 flag.
    class HWCE(nla):
      # nla_flags 这里只影响 encode. 对 decode 没有影响.
      nla_flags = NLA_F_NESTED
      nla_map = ((RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME, 'Name', 'asciiz'),
                 # 内核 rdma netlink 这里使用的是本机字节序
                 (RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_VALUE, 'Value', 'uint64'))

      # val: {'attrs': [('Name', 'listen_create_cnt'), ('Value', 0)]}
      @staticmethod
      def from_parsed(input: dict) -> tuple[str, int]:
        name: Optional[str] = None
        val: Optional[int] = None
        for k, v in input['attrs']:
          if k == 'Name':
            name = v
            continue
          if k == 'Value':
            val = v
            continue
        assert name is not None
        assert val is not None
        return name, val


_g_sock: Optional[AsyncNetlinkSocket] = None

def _get_sock():
  # use thread local?
  global _g_sock
  if _g_sock is not None:
    return _g_sock
  _g_sock = AsyncNetlinkSocket(family=NETLINK_RDMA)
  _g_sock.register_policy(RDMA_STATE_GET_TYPE, RdmaStatGetResp)
  return _g_sock


async def get_rdma_stat(devidx, portidx) -> dict[str, int]:
  sock = _get_sock()

  msg = RdmaStatGetMsg.new(devidx, portidx)
  req = NetlinkRequest(sock, msg)
  await req.send()
  ret: dict[str, int] = dict()
  async for resp in req.response():
    for k, v in resp['attrs']:
      if k == 'DevIndex':
        assert v == devidx
        continue
      if k == 'PortIndex':
        assert v == portidx
        continue
      if k != 'HWC':
        continue
      hwc = v
      for hwce_n, hwce_v in hwc['attrs']:
        if hwce_n != 'HWCE':
          continue
        ck, cv = RdmaStatGetResp.HWC.HWCE.from_parsed(hwce_v)
        assert ck not in ret
        ret[ck] = cv
    #print(resp)
  return ret


if __name__ == '__main__':
  import timeit

  loop = asyncio.get_event_loop()
  res = loop.run_until_complete(get_rdma_stat(3, 1))
  print(res)

  timeit_env = globals()
  timeit_env.update(locals())
  timeit_res = timeit.timeit('loop.run_until_complete(get_rdma_stat(0, 1))', number=1000, globals=timeit_env)
  print(timeit_res)
