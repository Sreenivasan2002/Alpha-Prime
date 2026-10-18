"""Diagnose why the app's strategy loses: run baseline on ALL days, then slice."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import harness as H
from strategies.app_current import Strategy

r = H.run(Strategy(), "all", slippage=0.0001)
t = r.trades.copy()
t.to_csv(os.path.join(os.path.dirname(__file__), "baseline_all_trades.csv"), index=False)
mk = H.Market(H.load_data())
print("BASELINE ALL 60 days:", {k: r.metrics[k] for k in ["trades","net","gross","costs","win_rate","profit_factor","up_days","down_days","t_stat","exits"]})

t["eh"] = pd.to_datetime(t.entry_time).dt.hour
print("\n1) BY ENTRY HOUR (avg net/trade, win%, n):")
print(t.groupby("eh").agg(n=("pnl","size"), avg=("pnl","mean"), win=("pnl", lambda x: (x>0).mean()*100), gross=("gross","sum")).round(0).to_string())

# gain at entry and path stats
rows = []
for _, x in t.iterrows():
    a = mk.arr[x.sym]; i = a["pos"][pd.Timestamp(x.entry_time)]
    j = a["pos"].get(pd.Timestamp(x.exit_time), i)
    pc = a["prev_close"][i]; gain_at_entry = (a["o"][i]/pc - 1)*100
    hi = a["h"][i:j+1].max(); lo = a["l"][i:j+1].min()
    d = a["date"][i]; day_idx = np.where(a["date"] == d)[0]
    fwd = {}
    for lbl, nb in [("f30m", 6), ("f60m", 12), ("f120m", 24)]:
        k = min(i + nb, day_idx[-1]); fwd[lbl] = (a["c"][k]/x.entry - 1)*100
    k_close = day_idx[-1]
    rows.append(dict(gain_at_entry=gain_at_entry, mfe=(hi/x.entry-1)*100, mae=(lo/x.entry-1)*100,
                     to_close=(a["c"][k_close]/x.entry-1)*100, dist_open=(x.entry/a["day_open"][i]-1)*100, **fwd))
p = pd.concat([t.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
p["gbin"] = pd.cut(p.gain_at_entry, [0,1.5,2,3,5,99])
print("\n2) BY HOW FAR THE STOCK HAD ALREADY RISEN AT ENTRY (vs prev close):")
print(p.groupby("gbin", observed=True).agg(n=("pnl","size"), avg=("pnl","mean"), win=("pnl", lambda x:(x>0).mean()*100), f60=("f60m","mean"), to_close=("to_close","mean")).round(2).to_string())
print("\n3) PRICE PATH AFTER ENTRY (avg %):", p[["f30m","f60m","f120m","to_close","mfe","mae"]].mean().round(3).to_dict())
print("   median MFE %:", round(p.mfe.median(),2), "| % trades that ever reached +1%:", round((p.mfe>=1).mean()*100,1), "| reached +2.5%:", round((p.mfe>=2.5).mean()*100,1))
print("   avg charges as % of position:", round((t.cost/(t.entry*t.qty)*100).mean(),3))

# 4) market direction dependence
n = mk.arr["NIFTY"]; nd = pd.DataFrame({"date": n["date"], "o": n["o"], "c": n["c"]}).groupby("date").agg(o=("o","first"), c=("c","last"))
nd["ret"] = (nd.c/nd.o-1)*100
dd = r.daily.to_frame("pnl").join(nd.ret)
print("\n4) STRATEGY DAY P&L vs NIFTY DAY RETURN: corr", round(dd.pnl.corr(dd.ret),2))
dd["mkt"] = pd.cut(dd.ret, [-9,-0.3,0.3,9], labels=["NIFTY down","flat","NIFTY up"])
print(dd.groupby("mkt", observed=True).pnl.agg(["count","sum","mean"]).round(0).to_string())

# 5) universe-wide test of the core idea (no TA): stocks up 1-5% at 10:00 -> forward return
res = []
for s in mk.symbols:
    a = mk.arr[s]
    for d in np.unique(a["date"]):
        idx = np.where(a["date"] == d)[0]
        m = a["mins"][idx]
        try:
            i10 = idx[np.where(m == 600)[0][0]]; i12 = idx[np.where(m == 720)[0][0]]; iend = idx[-1]
        except IndexError:
            continue
        pc = a["prev_close"][i10]
        if not pc or np.isnan(pc): continue
        g = (a["c"][i10-1]/pc-1)*100
        res.append(dict(g=g, to12=(a["o"][i12]/a["o"][i10]-1)*100, toclose=(a["c"][iend]/a["o"][i10]-1)*100))
u = pd.DataFrame(res); u["gbin"] = pd.cut(u.g, [-99,-3,-1,1,3,5,99])
print("\n5) ALL NIFTY-100 STOCK-DAYS: return from 10:00 open, grouped by gain at 09:55 (%; n):")
print(u.groupby("gbin", observed=True).agg(n=("g","size"), to12=("to12","mean"), toclose=("toclose","mean")).round(3).to_string())
