from .logger import log
from bfrt_controller.bfrt_grpc.client import BfruntimeRpcException

class PortManager:
    def __init__(self, target, gc, bfrt_info):
        self.log = log
        self.target = target
        self.gc = gc

        # get port table
        self.port_table = bfrt_info.table_get('$PORT')

        # Statistics table
        self.port_stats_table = bfrt_info.table_get('$PORT_STAT')

        # Front-panel port to dev port lookup table
        self.port_hdl_info_table = bfrt_info.table_get('$PORT_HDL_INFO')

        # dev port to FP port reverse lookup table (lazy initialization)
        self.dev_port_to_fp_port = None

        # List of active ports
        self.active_ports = []

    def get_dev_port(self, fp_port, lane):
        ''' Convert front-panel port to dev port.

            Keyword arguments:
                fp_port -- front panel port number
                lane -- lane number

            Returns:
                (success flag, dev port or error message)
        '''
        resp = self.port_hdl_info_table.entry_get(self.target, [
            self.port_hdl_info_table.make_key([
                self.gc.KeyTuple('$CONN_ID', fp_port),
                self.gc.KeyTuple('$CHNL_ID', lane)
            ])
        ], {'from_hw': False})

        try:
            dev_port = next(resp)[0].to_dict()['$DEV_PORT']
        except BfruntimeRpcException:
            return (False, 'Port {}/{} not found!'.format(fp_port, lane))
        else:
            return (True, dev_port)
        
    def get_fp_port(self, dev_port):
        ''' Get front panel port from dev port.

            Returns:
                (success flag, port or error message, lane or None)
        '''

        # If we haven't filled the reverse mapping dict yet, do so
        if self.dev_port_to_fp_port is None:
            self.dev_port_to_fp_port = {}

            # Get all ports
            resp = self.port_hdl_info_table.entry_get(self.target, [],
                                                      {'from_hw': False})

            # Fill in dictionary
            for v, k in resp:
                v = v.to_dict()
                k = k.to_dict()
                self.dev_port_to_fp_port[v['$DEV_PORT']] = (
                    k['$CONN_ID']['value'], k['$CHNL_ID']['value'])

        # Look up front panel port/lane from dev port
        if dev_port in self.dev_port_to_fp_port:
            return (True,) + self.dev_port_to_fp_port[dev_port]
        else:
            return (False, 'Invalid dev port {}'.format(dev_port), None)
        
    def add_port(self, front_panel_port, lane, speed, fec, an):
        ''' Add one port.

            Keyword arguments:
                front_panel_port -- front panel port number
                lane -- lane within the front panel port
                speed -- port bandwidth in Gbps, one of {10, 25, 40, 50, 100}
                fec -- forward error correction, one of {'none', 'fc', 'rs'}
                autoneg -- autonegotiation, one of {'default', 'enable', 'disable'}

            Returns:
                (success flag, None or error message)
        '''

        speed_conversion_table = {
            10: 'BF_SPEED_10G',
            25: 'BF_SPEED_25G',
            40: 'BF_SPEED_40G',
            50: 'BF_SPEED_50G',
            100: 'BF_SPEED_100G'
        }

        fec_conversion_table = {
            'none': 'BF_FEC_TYP_NONE',
            'fc': 'BF_FEC_TYP_FC',
            'rs': 'BF_FEC_TYP_RS'
        }

        an_conversion_table = {
            'default': 'PM_AN_DEFAULT',
            'enable': 'PM_AN_FORCE_ENABLE',
            'disable': 'PM_AN_FORCE_DISABLE'
        }

        success, dev_port = self.get_dev_port(front_panel_port, lane)
        if not success:
            return (False, dev_port)

        if dev_port in self.active_ports:
            msg = 'Port {}/{} already in active ports list'.format(
                front_panel_port, lane)
            self.log.warning(msg)
            return (False, msg)

        self.port_table.entry_add(self.target, [
            self.port_table.make_key([self.gc.KeyTuple('$DEV_PORT', dev_port)])
        ], [
            self.port_table.make_data([
                self.gc.DataTuple('$SPEED',
                                  str_val=speed_conversion_table[speed]),
                self.gc.DataTuple('$FEC', str_val=fec_conversion_table[fec]),
                self.gc.DataTuple('$AUTO_NEGOTIATION',
                                  str_val=an_conversion_table[an]),
                self.gc.DataTuple('$PORT_ENABLE', bool_val=True)
            ])
        ])
        self.log.info('Added port: {}/{} {}G {} {}'.format(
            front_panel_port, lane, speed, fec, an))

        self.active_ports.append(dev_port)

        return (True, None)
    
    def add_ports(self, port_list):
        ''' Add ports.

            Keyword arguments:
                port_list -- a list of tuples: (front panel port, lane, speed, FEC string, autoneg) where:
                 front_panel_port is the front panel port number
                 lane is the lane within the front panel port
                 speed is the port bandwidth in Gbps, one of {10, 25, 40, 50, 100}
                 fec (forward error correction) is one of {'none', 'fc', 'rs'}
                 autoneg (autonegotiation) is one of {'default', 'enable', 'disable'}

            Returns:
                (success flag, None or error message)
        '''

        for (front_panel_port, lane, speed, fec, an) in port_list:
            success, error_msg = self.add_port(front_panel_port, lane, speed,
                                               fec, an)
            if not success:
                return (False, error_msg)

        return (True, None)
    
    def remove_port(self, front_panel_port, lane):
        ''' Remove one port.

            Keyword arguments:
                front_panel_port -- front panel port number
                lane -- lane within the front panel port

            Returns:
                (success flag, None or error message)
        '''

        success, dev_port = self.get_dev_port(front_panel_port, lane)
        if not success:
            return (False, dev_port)

        # Remove on switch
        self.port_table.entry_del(self.target, [
            self.port_table.make_key([self.gc.KeyTuple('$DEV_PORT', dev_port)])
        ])

        self.log.info('Removed port: {}/{}'.format(front_panel_port, lane))

        # Remove from our local active port list
        self.active_ports.remove(dev_port)

        return (True, None)

