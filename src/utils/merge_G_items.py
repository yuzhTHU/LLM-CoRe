import logging
import nd2py as nd
from functools import reduce
from typing import List, Dict

_logger = logging.getLogger(__name__)


def merge_G_items(items: List[nd.Symbol]) -> List[nd.Symbol]:
    op = None
    F_items = []
    G_items = []
    for item in items:
        if isinstance(item, nd.Aggr):
            op = nd.Aggr
            G_items.append(item.operands[0])
        elif isinstance(item, nd.Rgga):
            # G_items.append(item.operands[0])
            # op = ???
            F_items.append(item)
        elif isinstance(item, nd.Readout):
            if op is None: 
                op = nd.Readout
            if item.operands[0].nettype in ['edge', 'scalar']:
                G_items.append(item.operands[0])
            elif item.operands[0].nettype == 'node':
                G_items.append(nd.sour(item.operands[0]))
            else:
                raise ValueError(f"Unknown nettype in Readout: {item.operands[0].nettype}") 
        elif isinstance(item, (nd.Sour, nd.Targ)):
            raise ValueError("Sour and Targ are not allowed in f_sst")
        else:
            F_items.append(item)
    items = F_items
    if len(G_items):
        items.append(op(reduce(lambda a, b: a + b, G_items)))
    return items
