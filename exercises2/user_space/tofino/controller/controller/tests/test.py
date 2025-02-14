
import logging
import sys
import os

sde = "~/bf-sde-9.13.0/"

sys.path.append(f"{sde}/install/lib/python3.10/site-packages/tofino")
sys.path.append(f"{sde}/install/lib/python3.10/site-packages/")
sys.path.append(f"{sde}/install/lib/python3.10/site-packages/p4testutils")

import ptf
from ptf import config
from ptf.thriftutils import *
import ptf.testutils as testutils
from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.client as gc
import grpc
sys.path.append(f"{sde}/../cap-manager.p4/src/controller")
# import Controller

class DropIfCapNotExistsTest(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        p4_name = "cap-manager"
        BfRuntimeTest.setUp(self, client_id, p4_name)

    def runTest(self):
        bfrt_info = self.interface.bfrt_info_get()
        cap_table = bfrt_info.table_get("cap_table")
        c = Controller()
        c.insert_capability(42, "10.0.0.1", "10.0.0.0.2")

        print(c.get_table("cap_table"))

        self.assertTrue(false, "sample fail")
