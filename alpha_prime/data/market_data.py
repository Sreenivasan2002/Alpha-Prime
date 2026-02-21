"""
Market Data Engine for Alpha-Prime
Provides unified access to market data via yfinance for historical data
and Groww API for live data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from loguru import logger
import pytz

from alpha_prime.core.config import settings
from alpha_prime.core.database import cache_set, cache_get


IST = pytz.timezone("Asia/Kolkata")


def to_nse_symbol(symbol: str) -> str:
    """Convert symbol to NSE yfinance format"""
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS"):
        return f"{symbol}.NS"
    return symbol


def to_bse_symbol(symbol: str) -> str:
    """Convert symbol to BSE yfinance format"""
    symbol = symbol.upper().strip()
    if not symbol.endswith(".BO"):
        return f"{symbol}.BO"
    return symbol


def clean_symbol(symbol: str) -> str:
    """Remove exchange suffix from symbol"""
    return symbol.replace(".NS", "").replace(".BO", "").upper().strip()


class MarketDataEngine:
    """Unified market data access layer"""

    def __init__(self):
        self._cache = {}

    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get the latest available price for a symbol"""
        try:
            nse_sym = to_nse_symbol(symbol)
            ticker = yf.Ticker(nse_sym)
            info = ticker.fast_info
            price = getattr(info, 'last_price', None)
            if price is None:
                price = getattr(info, 'previous_close', None)
            if price and price > 0:
                return round(float(price), 2)
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")

        # Fallback: try from history
        try:
            nse_sym = to_nse_symbol(symbol)
            hist = yf.download(nse_sym, period="1d", interval="1m", progress=False)
            if not hist.empty:
                # Handle both MultiIndex and regular columns
                if isinstance(hist.columns, pd.MultiIndex):
                    price = float(hist["Close"].iloc[-1].values[0])
                else:
                    price = float(hist["Close"].iloc[-1])
                return round(price, 2)
        except Exception as e:
            logger.warning(f"Fallback price fetch failed for {symbol}: {e}")

        return None

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols efficiently"""
        prices = {}
        nse_symbols = [to_nse_symbol(s) for s in symbols]
        try:
            tickers = yf.Tickers(" ".join(nse_symbols))
            for orig_sym, nse_sym in zip(symbols, nse_symbols):
                try:
                    ticker = tickers.tickers.get(nse_sym)
                    if ticker:
                        info = ticker.fast_info
                        price = getattr(info, 'last_price', None) or getattr(info, 'previous_close', None)
                        if price and price > 0:
                            prices[clean_symbol(orig_sym)] = round(float(price), 2)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Batch price fetch failed: {e}")
            # Fallback to individual lookups
            for sym in symbols:
                price = self.get_live_price(sym)
                if price:
                    prices[clean_symbol(sym)] = price
        return prices

    def get_historical_data(self, symbol: str, period: str = "6mo",
                           interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data.
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
        interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
        """
        try:
            nse_sym = to_nse_symbol(symbol)
            df = yf.download(nse_sym, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Remove duplicate column names
            df = df.loc[:, ~df.columns.duplicated()]

            df.index.name = "Date"
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_intraday_data(self, symbol: str, interval: str = "5m",
                         period: str = "5d") -> Optional[pd.DataFrame]:
        """Get intraday data. Uses 5d period by default to ensure enough candles
        even early in the trading day (market opens at 9:15, at 9:30 we only have 3 candles)."""
        return self.get_historical_data(symbol, period=period, interval=interval)

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed stock information"""
        try:
            nse_sym = to_nse_symbol(symbol)
            ticker = yf.Ticker(nse_sym)
            info = ticker.info
            return {
                "symbol": clean_symbol(symbol),
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "beta": info.get("beta", 0),
                "eps": info.get("trailingEps", 0),
                "book_value": info.get("bookValue", 0),
                "face_value": info.get("faceValue", 0),
                "previous_close": info.get("previousClose", 0),
                "open": info.get("open", 0),
                "day_high": info.get("dayHigh", 0),
                "day_low": info.get("dayLow", 0),
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None

    def get_nifty50_stocks(self) -> List[str]:
        """Get list of NIFTY 50 constituent stocks"""
        # Hardcoded NIFTY 50 stocks (updated periodically)
        return [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "HCLTECH",
            "ASIANPAINT", "SUNPHARMA", "TITAN", "WIPRO", "TMPV",
            "ULTRACEMCO", "NESTLEIND", "POWERGRID", "NTPC", "M&M",
            "ONGC", "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS",
            "TECHM", "BAJAJFINSV", "HDFCLIFE", "DIVISLAB", "DRREDDY",
            "SBILIFE", "GRASIM", "CIPLA", "COALINDIA", "BRITANNIA",
            "BPCL", "EICHERMOT", "INDUSINDBK", "TATACONSUM", "APOLLOHOSP",
            "HEROMOTOCO", "BAJAJ-AUTO", "UPL", "HINDALCO", "LTIM"
        ]

    def get_banknifty_stocks(self) -> List[str]:
        """Get list of Bank NIFTY constituent stocks"""
        return [
            "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
            "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB",
            "PNB", "AUBANK", "BANKBARODA"
        ]

    def get_nifty100_stocks(self) -> List[str]:
        """Get list of NIFTY 100 constituent stocks (NIFTY 50 + NIFTY Next 50)"""
        nifty50 = self.get_nifty50_stocks()
        next50 = [
            "ABB", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA",
            "BAJAJHLDNG", "BANKBARODA", "BEL", "BERGEPAINT", "BOSCHLTD",
            "CANBK", "CHOLAFIN", "COLPAL", "DLF", "DABUR",
            "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI", "INDHOTEL",
            "IOC", "IRCTC", "IRFC", "JIOFIN", "JSL",
            "LICI", "LUPIN", "MARICO", "MOTHERSON", "NAUKRI",
            "NHPC", "PFC", "PIDILITIND", "PNB", "POLYCAB",
            "RECLTD", "SBICARD", "SHREECEM", "SIEMENS", "SRF",
            "TATAPOWER", "TORNTPHARM", "TRENT", "UNITDSPR", "VEDL",
            "VBL", "YESBANK", "ZOMATO", "ZYDUSLIFE", "HAL"
        ]
        return nifty50 + next50

    def get_nifty500_stocks(self) -> List[str]:
        """Get a representative list of NIFTY 500 stocks (top ~200 by market cap)
        Full NIFTY 500 has 500 stocks - we fetch top 200 to keep response time reasonable."""
        nifty100 = self.get_nifty100_stocks()
        additional = [
            "AARTIIND", "ACC", "ALKEM", "APLLTD", "ASHOKLEY",
            "ASTRAL", "ATUL", "AUROPHARMA", "BALKRISIND", "BANDHANBNK",
            "BATAINDIA", "BHARATFORG", "BHEL", "BIOCON", "CANFINHOME",
            "CASTROLIND", "CENTRALBK", "CHAMBLFERT", "CLEAN", "COFORGE",
            "CONCOR", "CROMPTON", "CUB", "CUMMINSIND", "CYIENT",
            "DEEPAKNTR", "DELHIVERY", "DIXON", "ESCORTS", "EXIDEIND",
            "FEDERALBNK", "FORTIS", "GLENMARK", "GMRAIRPORT", "GNFC",
            "GSPL", "GUJGASLTD", "HDFCAMC", "HINDZINC", "HONAUT",
            "IDFCFIRSTB", "IEX", "IIFL", "INDUSTOWER", "INTELLECT",
            "IPCALAB", "JKCEMENT", "JUBLFOOD", "KANSAINER", "KEI",
            "LICHSGFIN", "LTF", "LTTS", "M&MFIN", "MANAPPURAM",
            "MFSL", "MGL", "MPHASIS", "MRF", "MUTHOOTFIN",
            "NAM-INDIA", "NATIONALUM", "NAVINFLUOR", "NMDC", "OBEROIRLTY",
            "OFSS", "PAGEIND", "PATANJALI", "PERSISTENT", "PETRONET",
            "PIIND", "PRESTIGE", "PVRINOX", "RAJESHEXPO", "RAMCOCEM",
            "RELAXO", "SAIL", "SBILIFE", "SHRIRAMFIN", "SONACOMS",
            "STARHEALTH", "SUNDARMFIN", "SUNDRMFAST", "SUPREMEIND", "SYNGENE",
            "TATACHEM", "TATACOMM", "TATAELXSI", "TATAINVEST", "THERMAX",
            "TIINDIA", "TIMKEN", "TORNTPOWER", "TVSMOTOR", "UBL",
            "UNIONBANK", "UPL", "VOLTAS", "WHIRLPOOL", "ZEEL"
        ]
        # Deduplicate
        seen = set(nifty100)
        extras = [s for s in additional if s not in seen]
        return nifty100 + extras

    def get_nifty_midcap100_stocks(self) -> List[str]:
        """Get list of NIFTY Midcap 100 constituent stocks"""
        return [
            "ABB", "ABCAPITAL", "ACC", "ALKEM", "ASHOKLEY",
            "ASTRAL", "ATUL", "AUBANK", "BALKRISIND", "BANDHANBNK",
            "BATAINDIA", "BEL", "BHARATFORG", "BHEL", "BIOCON",
            "CANFINHOME", "CENTRALBK", "CHOLAFIN", "CLEAN", "COFORGE",
            "COLPAL", "CONCOR", "CROMPTON", "CUMMINSIND", "CYIENT",
            "DABUR", "DEEPAKNTR", "DELHIVERY", "DIXON", "DLF",
            "ESCORTS", "EXIDEIND", "FEDERALBNK", "FORTIS", "GAIL",
            "GLENMARK", "GMRAIRPORT", "GNFC", "GODREJCP", "GSPL",
            "GUJGASLTD", "HAL", "HAVELLS", "HDFCAMC", "HONAUT",
            "ICICIPRULI", "IDFCFIRSTB", "IEX", "IIFL", "INDHOTEL",
            "INDUSTOWER", "INTELLECT", "IOC", "IPCALAB", "IRCTC",
            "IRFC", "JIOFIN", "JKCEMENT", "JSL", "JUBLFOOD",
            "KANSAINER", "KEI", "LICI", "LICHSGFIN", "LTF",
            "LTTS", "LUPIN", "M&MFIN", "MANAPPURAM", "MARICO",
            "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF",
            "MUTHOOTFIN", "NAM-INDIA", "NATIONALUM", "NAUKRI", "NAVINFLUOR",
            "NHPC", "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND",
            "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND",
            "PNB", "POLYCAB", "PRESTIGE", "PVRINOX", "RECLTD",
            "SAIL", "SBICARD", "SHREECEM", "SHRIRAMFIN", "SIEMENS",
        ]

    def get_nifty_smallcap100_stocks(self) -> List[str]:
        """Get list of NIFTY Smallcap 100 constituent stocks"""
        return [
            "AARTIIND", "AETHER", "AFFLE", "AJANTPHARM", "ALOKINDS",
            "ANGELONE", "ANURAS", "APTUS", "ASTRAZEN", "ATUL",
            "BASF", "BAYERCROP", "BDL", "BIKAJI", "BLS",
            "BSE", "CAMPUS", "CAMS", "CARBORUNIV", "CASTROLIND",
            "CDSL", "CESC", "CHAMBLFERT", "CHALET", "COCHINSHIP",
            "CRAFTSMAN", "CYIENT", "DATAPATTNS", "DCMSHRIRAM", "DEVYANI",
            "DOMS", "EASEMYTRIP", "ELGIEQUIP", "EMAMILTD", "ENDURANCE",
            "EQUITASBNK", "FINEORG", "FIVESTAR", "FLUOROCHEM", "GLAXO",
            "GRINDWELL", "GSFC", "HAPPSTMNDS", "HEG", "HINDPETRO",
            "HOMEFIRST", "HUNTSMANIN", "IBULHSGFIN", "IDFC", "INDIGOPNTS",
            "JBCHEPHARM", "JBMA", "JWL", "KAJARIACER", "KALPATPOWR",
            "KFINTECH", "KSB", "LAXMIMACH", "LEMONTREE", "LLOYDSME",
            "LUXIND", "MAHABANK", "MAHLIFE", "MAPMYINDIA", "MASTEK",
            "METROPOLIS", "NATCOPHARM", "OLECTRA", "PGHH", "PHOENIXLTD",
            "PNBHOUSING", "POWERINDIA", "PPLPHARMA", "PRSMJOHNSN", "RADICO",
            "RBLBANK", "REDINGTON", "RITES", "ROUTE", "SAPPHIRE",
            "SCHNEIDER", "SJVN", "SKFINDIA", "SOLARINDS", "SONATSOFTW",
            "SPARC", "SUMICHEM", "SUNTV", "TATAINVEST", "TATVA",
            "TEAMLEASE", "TECHNOE", "TRIDENT", "TRITURBINE", "TTML",
            "TV18BRDCST", "UTIAMC", "VINATIORGA", "VMART", "WELCORP",
        ]

    def get_nifty_total_market_stocks(self) -> List[str]:
        """Get a broad list representing Nifty Total Market Index
        (combines NIFTY 500 representative list - largest coverage available)"""
        return self.get_nifty500_stocks()

    def get_index_stocks(self, index_name: str) -> List[str]:
        """Get stock list for a given index name"""
        index_map = {
            "NIFTY 50": self.get_nifty50_stocks,
            "NIFTY 100": self.get_nifty100_stocks,
            "NIFTY 500": self.get_nifty500_stocks,
            "NIFTY Midcap 100": self.get_nifty_midcap100_stocks,
            "NIFTY Smallcap 100": self.get_nifty_smallcap100_stocks,
            "Nifty Total Market": self.get_nifty_total_market_stocks,
            "Bank NIFTY": self.get_banknifty_stocks,
        }
        getter = index_map.get(index_name, self.get_nifty50_stocks)
        return getter()


def is_market_open() -> bool:
    """Check if Indian market is currently open"""
    now = datetime.now(IST)

    # Weekend check
    if now.weekday() >= 5:
        return False

    market_open = now.replace(
        hour=settings.market.market_open_hour,
        minute=settings.market.market_open_minute,
        second=0, microsecond=0
    )
    market_close = now.replace(
        hour=settings.market.market_close_hour,
        minute=settings.market.market_close_minute,
        second=0, microsecond=0
    )

    return market_open <= now <= market_close


def is_pre_market() -> bool:
    """Check if we're in pre-market session"""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False

    pre_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

    return pre_open <= now < market_open


def time_to_market_open() -> Optional[timedelta]:
    """Get time until market opens"""
    now = datetime.now(IST)
    if is_market_open():
        return timedelta(0)

    # Next market open
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now >= market_open:
        # Next business day
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        market_open = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
    return market_open - now


# Singleton instance
market_data = MarketDataEngine()
