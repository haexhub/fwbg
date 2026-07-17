"""
IG Markets Symbol und Timeframe Mappings.

Enthält alle broker-spezifischen Mappings für IG Markets:
- Symbol -> Epic (IG Instrument Identifier)
- Symbol -> yfinance Ticker (Fallback)
- Symbol -> Point Value (für Pip-Berechnung)
- Timeframe -> Resolution (IG API Format)
- Timeframe -> yfinance Interval (Fallback)
"""
from typing import Dict

from fwbg_sdk import Symbol, Timeframe


# Symbol -> Epic Mapping für deutsche IG Accounts
# EPICs sind regional unterschiedlich!
# Forex: MINI.IP = Mini CFD (empfohlen), CFD.IP = Standard CFD
# TODAY.IP = Daily Cash (täglicher Ablauf, NICHT für Trading empfohlen!)
SYMBOL_TO_EPIC: Dict[Symbol, str] = {
    # Indizes CFDs
    Symbol.FTSE100: "IX.D.FTSE.DAILY.IP",
    Symbol.DOW30: "IX.D.DOW.DAILY.IP",
    Symbol.NAS100: "IX.D.NASDAQ.DAILY.IP",
    Symbol.DAX: "IX.D.DAX.DAILY.IP",
    Symbol.SPX500: "IX.D.SPTRD.DAILY.IP",
    # Forex Majors (Standard CFD - 100k Kontrakt)
    Symbol.EURUSD: "CS.D.EURUSD.CFD.IP",
    Symbol.GBPUSD: "CS.D.GBPUSD.CFD.IP",
    Symbol.USDJPY: "CS.D.USDJPY.CFD.IP",
    Symbol.USDCHF: "CS.D.USDCHF.CFD.IP",
    Symbol.USDCAD: "CS.D.USDCAD.CFD.IP",
    Symbol.AUDUSD: "CS.D.AUDUSD.CFD.IP",
    Symbol.NZDUSD: "CS.D.NZDUSD.CFD.IP",
    # Forex Crosses EUR (Standard CFD)
    Symbol.EURCAD: "CS.D.EURCAD.CFD.IP",
    Symbol.EURCHF: "CS.D.EURCHF.CFD.IP",
    Symbol.EURGBP: "CS.D.EURGBP.CFD.IP",
    Symbol.EURJPY: "CS.D.EURJPY.CFD.IP",
    Symbol.EURAUD: "CS.D.EURAUD.CFD.IP",
    Symbol.EURNZD: "CS.D.EURNZD.CFD.IP",
    # Forex Crosses GBP (Standard CFD)
    Symbol.GBPAUD: "CS.D.GBPAUD.CFD.IP",
    Symbol.GBPCAD: "CS.D.GBPCAD.CFD.IP",
    Symbol.GBPCHF: "CS.D.GBPCHF.CFD.IP",
    Symbol.GBPJPY: "CS.D.GBPJPY.CFD.IP",
    Symbol.GBPNZD: "CS.D.GBPNZD.CFD.IP",
    # Forex Crosses AUD (Standard CFD)
    Symbol.AUDCAD: "CS.D.AUDCAD.CFD.IP",
    Symbol.AUDCHF: "CS.D.AUDCHF.CFD.IP",
    Symbol.AUDJPY: "CS.D.AUDJPY.CFD.IP",
    Symbol.AUDNZD: "CS.D.AUDNZD.CFD.IP",
    # Forex Crosses NZD (Standard CFD)
    Symbol.NZDCAD: "CS.D.NZDCAD.CFD.IP",
    Symbol.NZDCHF: "CS.D.NZDCHF.CFD.IP",
    Symbol.NZDJPY: "CS.D.NZDJPY.CFD.IP",
    # Forex Crosses CAD/CHF (Standard CFD)
    Symbol.CADCHF: "CS.D.CADCHF.CFD.IP",
    Symbol.CADJPY: "CS.D.CADJPY.CFD.IP",
    Symbol.CHFJPY: "CS.D.CHFJPY.CFD.IP",
    # Commodities
    Symbol.XAUUSD: "CS.D.CFDGOLD.CFD.IP",
    Symbol.XAGUSD: "CS.D.CFDSILVER.CFD.IP",
    Symbol.BRENT: "CC.D.LCO.UNC.IP",
    Symbol.WTI: "CC.D.CL.UNC.IP",
    # Crypto
    Symbol.BTCUSD: "CS.D.BITCOIN.CFD.IP",
    Symbol.ETHUSD: "CS.D.ETHUSD.CFD.IP",
}

# yfinance Ticker Mapping (Fallback bei Rate Limiting)
SYMBOL_TO_YFINANCE: Dict[Symbol, str] = {
    # Forex Majors
    Symbol.EURUSD: "EURUSD=X",
    Symbol.GBPUSD: "GBPUSD=X",
    Symbol.USDJPY: "USDJPY=X",
    Symbol.USDCHF: "USDCHF=X",
    Symbol.USDCAD: "USDCAD=X",
    Symbol.AUDUSD: "AUDUSD=X",
    Symbol.NZDUSD: "NZDUSD=X",
    # Forex Crosses
    Symbol.EURCAD: "EURCAD=X",
    Symbol.EURCHF: "EURCHF=X",
    Symbol.EURGBP: "EURGBP=X",
    Symbol.EURJPY: "EURJPY=X",
    Symbol.EURAUD: "EURAUD=X",
    Symbol.EURNZD: "EURNZD=X",
    Symbol.GBPAUD: "GBPAUD=X",
    Symbol.GBPCAD: "GBPCAD=X",
    Symbol.GBPCHF: "GBPCHF=X",
    Symbol.GBPJPY: "GBPJPY=X",
    Symbol.GBPNZD: "GBPNZD=X",
    Symbol.AUDCAD: "AUDCAD=X",
    Symbol.AUDCHF: "AUDCHF=X",
    Symbol.AUDJPY: "AUDJPY=X",
    Symbol.AUDNZD: "AUDNZD=X",
    Symbol.NZDCAD: "NZDCAD=X",
    Symbol.NZDCHF: "NZDCHF=X",
    Symbol.NZDJPY: "NZDJPY=X",
    Symbol.CADCHF: "CADCHF=X",
    Symbol.CADJPY: "CADJPY=X",
    Symbol.CHFJPY: "CHFJPY=X",
    # Indizes
    Symbol.FTSE100: "^FTSE",
    Symbol.DOW30: "^DJI",
    Symbol.NAS100: "^NDX",
    Symbol.DAX: "^GDAXI",
    Symbol.SPX500: "^GSPC",
    # Commodities
    Symbol.XAUUSD: "GC=F",
    Symbol.XAGUSD: "SI=F",
    Symbol.BRENT: "BZ=F",
    Symbol.WTI: "CL=F",
    # Crypto
    Symbol.BTCUSD: "BTC-USD",
    Symbol.ETHUSD: "ETH-USD",
}

# Point Value für Pips-Berechnung
SYMBOL_POINT_VALUE: Dict[Symbol, float] = {
    # Forex Majors
    Symbol.EURUSD: 0.0001,
    Symbol.GBPUSD: 0.0001,
    Symbol.USDJPY: 0.01,
    Symbol.USDCHF: 0.0001,
    Symbol.USDCAD: 0.0001,
    Symbol.AUDUSD: 0.0001,
    Symbol.NZDUSD: 0.0001,
    # Forex Crosses EUR
    Symbol.EURCAD: 0.0001,
    Symbol.EURCHF: 0.0001,
    Symbol.EURGBP: 0.0001,
    Symbol.EURJPY: 0.01,
    Symbol.EURAUD: 0.0001,
    Symbol.EURNZD: 0.0001,
    # Forex Crosses GBP
    Symbol.GBPAUD: 0.0001,
    Symbol.GBPCAD: 0.0001,
    Symbol.GBPCHF: 0.0001,
    Symbol.GBPJPY: 0.01,
    Symbol.GBPNZD: 0.0001,
    # Forex Crosses AUD
    Symbol.AUDCAD: 0.0001,
    Symbol.AUDCHF: 0.0001,
    Symbol.AUDJPY: 0.01,
    Symbol.AUDNZD: 0.0001,
    # Forex Crosses NZD
    Symbol.NZDCAD: 0.0001,
    Symbol.NZDCHF: 0.0001,
    Symbol.NZDJPY: 0.01,
    # Forex Crosses CAD/CHF
    Symbol.CADCHF: 0.0001,
    Symbol.CADJPY: 0.01,
    Symbol.CHFJPY: 0.01,
    # Commodities
    Symbol.XAUUSD: 0.01,
    Symbol.XAGUSD: 0.001,
    Symbol.BRENT: 0.01,
    Symbol.WTI: 0.01,
    # Indizes
    Symbol.DAX: 1.0,
    Symbol.DOW30: 1.0,
    Symbol.NAS100: 1.0,
    Symbol.SPX500: 0.1,
    Symbol.FTSE100: 1.0,
    # Crypto
    Symbol.BTCUSD: 0.01,
    Symbol.ETHUSD: 0.01,
}

# IG Resolution Mapping (IG API format)
# Gültige Werte: SECOND, MINUTE, MINUTE_2, MINUTE_3, MINUTE_5, MINUTE_10,
#                MINUTE_15, MINUTE_30, HOUR, HOUR_2, HOUR_3, HOUR_4, DAY, WEEK, MONTH
TIMEFRAME_TO_RESOLUTION: Dict[Timeframe, str] = {
    Timeframe.M1: "MINUTE",
    Timeframe.M5: "MINUTE_5",
    Timeframe.M15: "MINUTE_15",
    Timeframe.M30: "MINUTE_30",
    Timeframe.H1: "HOUR",
    Timeframe.H2: "HOUR_2",
    Timeframe.H4: "HOUR_4",
    Timeframe.D1: "DAY",
    Timeframe.W1: "WEEK",
}

# yfinance Interval Mapping
# H2 bewusst nicht gemappt: yfinance kennt kein 2-Stunden-Intervall.
TIMEFRAME_TO_YF_INTERVAL: Dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
    Timeframe.W1: "1wk",
}
