import Map
import pandas as pd


def Search():
    rd = []
    sn = [str(Map.id)]

    def rd_append(k):
        if Map.edge3list[k].properties2.目标节点 not in rd:
            rd.append(Map.edge3list[k].properties2.目标节点)

    def sn_append(k):
        if Map.edge3list[k].properties2.目标节点 not in sn:
            sn.append(Map.edge3list[k].properties2.目标节点)

    while len(sn) > 0:
        sntop = sn[0]
        for i in range(len(Map.edge3list)):
            if Map.edge3list[i].properties2.源节点 == sntop:
                if Map.edge3list[i].type == "Q-A":
                    rd_append(i)
                    sn_append(i)
                if Map.edge3list[i].type == "duplicate":
                    df = pd.read_sql('select * from posts where id = ' + str(Map.edge3list[i].properties2.目标节点) + ';',
                                     con=Map.conn)
                    if df["PostTypeId"] == 1:
                        sn_append(i)
                    if df["PostTypeId"] == 2:
                        rd_append(i)
                        sn_append(i)
                if Map.edge3list[i].type == "existing":
                    rd_append(i)
        sn.pop()

    return rd
