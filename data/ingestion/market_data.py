"""
Market Data Manager
====================
Real-time market data via Binance WebSocket streams + REST fallback.
Provides OHLCV candles, order book snapshots, and live ticker prices.
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import websocket
import httpx

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """Single OHLCV candle."""
    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    is_closed: bool

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.open_time / 1000, tz=timezone.utc)


@dataclass
class TickerUpdate:
    """Real-time ticker price update."""
    symbol: str
    price: float
    timestamp: float


@dataclass
class OrderBookSnapshot:
    """Top-of-book snapshot."""
    symbol: str
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    spread: float
    spread_pct: float
    timestamp: float


class MarketDataManager:
    """
    Manages real-time market data streams.
    
    Features:
    - WebSocket streams for live klines and tickers
    - REST fallback for historical candles
    - Candle buffer per symbol (configurable depth)
    - Callbacks for strategy signals
    - Auto-reconnection on disconnect
    """

    def __init__(self, config, max_candles: int = 200):
        self._config = config
        self._max_candles = max_candles
        self._base_url = config.exchange.base_url
        self._ws_base = "wss://stream.binance.com:9443/ws"

        # Data stores
        self._candles: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_candles))
        )
        self._tickers: Dict[str, TickerUpdate] = {}
        self._orderbooks: Dict[str, OrderBookSnapshot] = {}

        # Callbacks
        self._candle_callbacks: List[Callable] = []
        self._ticker_callbacks: List[Callable] = []

        # WebSocket state
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60

        # Rate limiting for REST
        self._last_rest_call = 0
        self._rest_min_interval = 0.1  # 100ms between REST calls

        logger.info(
            f"MarketDataManager initialized | "
            f"Symbols: {len(config.all_trading_pairs)} | "
            f"Buffer: {max_candles} candles"
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_candle(self, callback: Callable[[Candle], None]):
        """Register callback for new candle data."""
        self._candle_callbacks.append(callback)

    def on_ticker(self, callback: Callable[[TickerUpdate], None]):
        """Register callback for ticker updates."""
        self._ticker_callbacks.append(callback)

    def _notify_candle(self, candle: Candle):
        for cb in self._candle_callbacks:
            try:
                cb(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")

    def _notify_ticker(self, ticker: TickerUpdate):
        for cb in self._ticker_callbacks:
            try:
                cb(ticker)
            except Exception as e:
                logger.error(f"Ticker callback error: {e}")

    # =========================================================================
    # REST API - Historical Data
    # =========================================================================

    def fetch_candles(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> List[Candle]:
        """Fetch historical candles via REST API."""
        self._rate_limit_rest()
        url = f"{self._base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            candles = []
            for k in data:
                candle = Candle(
                    symbol=symbol,
                    interval=interval,
                    open_time=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    close_time=k[6],
                    quote_volume=float(k[7]),
                    trades=k[8],
                    is_closed=True,
                )
                candles.append(candle)
                # Store in buffer
                self._candles[symbol][interval].append(candle)

            logger.info(f"Fetched {len(candles)} {interval} candles for {symbol}")
            return candles

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return []

    def fetch_orderbook(self, symbol: str, limit: int = 5) -> Optional[OrderBookSnapshot]:
        """Fetch current order book top."""
        self._rate_limit_rest()
        url = f"{self._base_url}/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if data["bids"] and data["asks"]:
                best_bid = float(data["bids"][0][0])
                best_ask = float(data["asks"][0][0])
                spread = best_ask - best_bid
                mid = (best_bid + best_ask) / 2

                snapshot = OrderBookSnapshot(
                    symbol=symbol,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_qty=float(data["bids"][0][1]),
                    ask_qty=float(data["asks"][0][1]),
                    spread=spread,
                    spread_pct=(spread / mid * 100) if mid > 0 else 0,
                    timestamp=time.time(),
                )
                self._orderbooks[symbol] = snapshot
                return snapshot

        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
        return None

    def preload_candles(self, symbols: List[str] = None, interval: str = "1m", limit: int = 100):
        """Preload historical candles for all or specified symbols."""
        symbols = symbols or self._config.all_trading_pairs
        loaded = 0
        for symbol in symbols:
            candles = self.fetch_candles(symbol, interval, limit)
            if candles:
                loaded += 1
            time.sleep(0.1)  # Rate limiting
        logger.info(f"Preloaded candles for {loaded}/{len(symbols)} symbols")

    # =========================================================================
    # WEBSOCKET - Real-time Streams
    # =========================================================================

    def start_streams(self, symbols: List[str] = None, intervals: List[str] = None):
        """Start WebSocket streams for live data."""
        symbols = symbols or self._config.all_trading_pairs
        intervals = intervals or ["1m"]

        # Build stream names
        streams = []
        for symbol in symbols:
            s = symbol.lower()
            # Kline streams
            for interval in intervals:
                streams.append(f"{s}@kline_{interval}")
            # Mini ticker stream
            streams.append(f"{s}@miniTicker")

        stream_url = f"{self._ws_base}/{'/'.join(streams)}" if len(streams) == 1 else \
            f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        self._running = True
        self._ws_thread = threading.Thread(
            target=self._run_websocket, args=(stream_url,), daemon=True
        )
        self._ws_thread.start()
        logger.info(f"WebSocket streams starting | {len(streams)} streams for {len(symbols)} symbols")

    def stop_streams(self):
        """Stop all WebSocket streams."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
        logger.info("WebSocket streams stopped")

    def _run_websocket(self, url: str):
        """WebSocket connection loop with auto-reconnect."""
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                    on_open=self._on_ws_open,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    def _on_ws_open(self, ws):
        logger.info("WebSocket connected")
        self._reconnect_delay = 1  # Reset on successful connect

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)

            # Combined stream format
            if "stream" in data:
                stream_name = data["stream"]
                payload = data["data"]
            else:
                stream_name = ""
                payload = data

            if "kline" in stream_name or "e" in payload and payload.get("e") == "kline":
                self._handle_kline(payload)
            elif "miniTicker" in stream_name or payload.get("e") == "24hrMiniTicker":
                self._handle_ticker(payload)

        except Exception as e:
            logger.error(f"WebSocket message parse error: {e}")

    def _on_ws_error(self, ws, error):
        logger.warning(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")

    def _handle_kline(self, data):
        """Process incoming kline/candle data."""
        k = data.get("k", data)
        candle = Candle(
            symbol=k["s"],
            interval=k["i"],
            open_time=k["t"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=k["T"],
            quote_volume=float(k["q"]),
            trades=k["n"],
            is_closed=k["x"],
        )

        # Update buffer
        buf = self._candles[candle.symbol][candle.interval]
        if buf and not candle.is_closed:
            # Update current candle in-place
            if buf[-1].open_time == candle.open_time:
                buf[-1] = candle
            else:
                buf.append(candle)
        elif candle.is_closed:
            buf.append(candle)
            self._notify_candle(candle)

        # Also update ticker from candle close
        self._tickers[candle.symbol] = TickerUpdate(
            symbol=candle.symbol, price=candle.close, timestamp=time.time()
        )

    def _handle_ticker(self, data):
        """Process incoming mini ticker data."""
        ticker = TickerUpdate(
            symbol=data["s"],
            price=float(data["c"]),
            timestamp=time.time(),
        )
        self._tickers[ticker.symbol] = ticker
        self._notify_ticker(ticker)

    # =========================================================================
    # DATA ACCESS
    # =========================================================================

    def get_candles(self, symbol: str, interval: str = "1m") -> List[Candle]:
        """Get buffered candles for a symbol."""
        return list(self._candles[symbol][interval])

    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        ticker = self._tickers.get(symbol)
        return ticker.price if ticker else None

    def get_prices(self) -> Dict[str, float]:
        """Get all latest prices."""
        return {s: t.price for s, t in self._tickers.items()}

    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get cached order book snapshot."""
        return self._orderbooks.get(symbol)

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current spread percentage for a symbol."""
        ob = self._orderbooks.get(symbol)
        return ob.spread_pct if ob else None

    def get_data_status(self) -> dict:
        """Get status of market data feeds."""
        return {
            "streaming": self._running,
            "symbols_with_candles": len(self._candles),
            "symbols_with_prices": len(self._tickers),
            "symbols_with_orderbooks": len(self._orderbooks),
            "candle_buffers": {
                symbol: {iv: len(buf) for iv, buf in intervals.items()}
                for symbol, intervals in self._candles.items()
            },
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _rate_limit_rest(self):
        """Simple rate limiting for REST calls."""
        elapsed = time.time() - self._last_rest_call
        if elapsed < self._rest_min_interval:
            time.sleep(self._rest_min_interval - elapsed)
        self._last_rest_call = time.time()
