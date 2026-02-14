"""
Exchange Adapter - Binance
==========================
Clean, single-responsibility exchange integration.
Handles: connection, authentication, order submission, balance queries.
Does NOT handle: strategy decisions, risk checks, position management.
"""

import os
import time
import hmac
import hashlib
import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Standardized order result."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    filled_quantity: float
    avg_fill_price: float
    commission: float
    commission_asset: str
    timestamp: str
    raw_response: dict


@dataclass
class Balance:
    """Account balance for a single asset."""
    asset: str
    free: float
    locked: float

    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass
class TickerPrice:
    """Current price for a symbol."""
    symbol: str
    price: float
    timestamp: float


class BinanceAdapter:
    """
    Binance exchange adapter.

    Responsibilities:
    - API authentication and request signing
    - Order placement (limit, market)
    - Balance queries
    - Price data retrieval
    - Rate limit tracking

    NOT responsible for:
    - Trade decisions
    - Risk management
    - Position tracking
    """

    def __init__(self, config, paper_mode: bool = True):
        self.config = config.exchange
        self.api_key = os.environ.get("BINANCE_API_KEY", "")
        self.api_secret = os.environ.get("BINANCE_SECRET_KEY", "")
        self.base_url = config.exchange.base_url
        self.paper_mode = paper_mode

        # Rate limiting
        self._request_timestamps: List[float] = []
        self._order_timestamps: List[float] = []

        # HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=10.0,
            headers={
                "X-MBX-APIKEY": self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        if not self.api_key or not self.api_secret:
            logger.warning("Binance API keys not set - only public endpoints available")

        logger.info(
            f"BinanceAdapter initialized | URL: {self.base_url} | "
            f"Paper: {self.paper_mode} | Keys: {'SET' if self.api_key else 'MISSING'}"
        )

    # =========================================================================
    # PUBLIC DATA (no auth required)
    # =========================================================================

    def get_price(self, symbol: str) -> Optional[TickerPrice]:
        """Get current price for a symbol."""
        try:
            self._check_rate_limit()
            resp = self._client.get("/api/v3/ticker/price", params={"symbol": symbol})
            resp.raise_for_status()
            data = resp.json()
            return TickerPrice(
                symbol=data["symbol"],
                price=float(data["price"]),
                timestamp=time.time(),
            )
        except Exception as e:
            logger.error(f"Price fetch failed for {symbol}: {e}")
            return None

    def get_prices(self, symbols: List[str]) -> Dict[str, TickerPrice]:
        """Get current prices for multiple symbols."""
        try:
            self._check_rate_limit()
            resp = self._client.get("/api/v3/ticker/price")
            resp.raise_for_status()
            data = resp.json()
            prices = {}
            symbol_set = set(symbols)
            for item in data:
                if item["symbol"] in symbol_set:
                    prices[item["symbol"]] = TickerPrice(
                        symbol=item["symbol"],
                        price=float(item["price"]),
                        timestamp=time.time(),
                    )
            return prices
        except Exception as e:
            logger.error(f"Bulk price fetch failed: {e}")
            return {}

    def get_klines(
        self, symbol: str, interval: str = "1m", limit: int = 200
    ) -> List[dict]:
        """Get candlestick data."""
        try:
            self._check_rate_limit()
            resp = self._client.get(
                "/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
            resp.raise_for_status()
            raw = resp.json()
            return [
                {
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                    "quote_volume": float(k[7]),
                    "trades": k[8],
                }
                for k in raw
            ]
        except Exception as e:
            logger.error(f"Klines fetch failed for {symbol}: {e}")
            return []

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[dict]:
        """Get order book depth."""
        try:
            self._check_rate_limit()
            resp = self._client.get(
                "/api/v3/depth", params={"symbol": symbol, "limit": limit}
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "bids": [(float(p), float(q)) for p, q in data["bids"]],
                "asks": [(float(p), float(q)) for p, q in data["asks"]],
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Orderbook fetch failed for {symbol}: {e}")
            return None

    # =========================================================================
    # ACCOUNT DATA (auth required)
    # =========================================================================

    def get_balances(self) -> Dict[str, Balance]:
        """Get all account balances."""
        if self.paper_mode:
            logger.debug("Paper mode - returning empty balances")
            return {}

        try:
            data = self._signed_request("GET", "/api/v3/account")
            balances = {}
            for b in data.get("balances", []):
                free = float(b["free"])
                locked = float(b["locked"])
                if free > 0 or locked > 0:
                    balances[b["asset"]] = Balance(
                        asset=b["asset"], free=free, locked=locked
                    )
            return balances
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return {}

    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        balances = self.get_balances()
        usdt = balances.get("USDT")
        return usdt.free if usdt else 0.0

    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Optional[OrderResult]:
        """Place a market order."""
        if self.paper_mode:
            return self._paper_order(symbol, side, quantity, "MARKET")

        try:
            self._check_order_rate_limit()
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": self._format_quantity(symbol, quantity),
            }
            data = self._signed_request("POST", "/api/v3/order", params)
            return self._parse_order_response(data)
        except Exception as e:
            logger.error(f"Market order failed: {symbol} {side} {quantity} - {e}")
            return None

    def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Optional[OrderResult]:
        """Place a limit order."""
        if self.paper_mode:
            return self._paper_order(symbol, side, quantity, "LIMIT", price)

        try:
            self._check_order_rate_limit()
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": self._format_quantity(symbol, quantity),
                "price": self._format_price(symbol, price),
            }
            data = self._signed_request("POST", "/api/v3/order", params)
            return self._parse_order_response(data)
        except Exception as e:
            logger.error(f"Limit order failed: {symbol} {side} {quantity}@{price} - {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        if self.paper_mode:
            logger.info(f"[PAPER] Order {order_id} cancelled")
            return True

        try:
            self._signed_request(
                "DELETE", "/api/v3/order",
                {"symbol": symbol, "orderId": order_id}
            )
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {order_id} - {e}")
            return False

    # =========================================================================
    # INTERNALS
    # =========================================================================

    def _signed_request(self, method: str, path: str, params: dict = None) -> dict:
        """Make a signed API request."""
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        self._check_rate_limit()

        if method == "GET":
            resp = self._client.get(path, params=params)
        elif method == "POST":
            resp = self._client.post(path, data=params)
        elif method == "DELETE":
            resp = self._client.delete(path, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")

        resp.raise_for_status()
        return resp.json()

    def _paper_order(
        self, symbol: str, side: str, quantity: float,
        order_type: str, price: float = None
    ) -> OrderResult:
        """Simulate an order in paper mode."""
        if price is None:
            ticker = self.get_price(symbol)
            price = ticker.price if ticker else 0.0

        fee_rate = self.config.spot_taker_pct / 100 if order_type == "MARKET" else self.config.spot_maker_pct / 100
        # Note: fee_rate comes from exchange config, not from self.config directly
        # This is a simplified paper order - the execution engine handles fee tracking

        result = OrderResult(
            order_id=f"PAPER_{int(time.time() * 1000)}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status="FILLED",
            filled_quantity=quantity,
            avg_fill_price=price,
            commission=quantity * price * 0.001,  # Approximate 0.1%
            commission_asset="USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_response={},
        )

        logger.info(
            f"[PAPER] {side} {quantity} {symbol} @ ${price:.4f} "
            f"(${quantity * price:.2f})"
        )
        return result

    def _parse_order_response(self, data: dict) -> OrderResult:
        """Parse Binance order response into standardized format."""
        fills = data.get("fills", [])
        total_commission = sum(float(f.get("commission", 0)) for f in fills)
        commission_asset = fills[0].get("commissionAsset", "USDT") if fills else "USDT"

        return OrderResult(
            order_id=str(data["orderId"]),
            symbol=data["symbol"],
            side=data["side"].lower(),
            quantity=float(data["origQty"]),
            price=float(data.get("price", 0)),
            status=data["status"],
            filled_quantity=float(data.get("executedQty", 0)),
            avg_fill_price=float(data.get("cummulativeQuoteQty", 0)) / max(float(data.get("executedQty", 1)), 1e-10),
            commission=total_commission,
            commission_asset=commission_asset,
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_response=data,
        )

    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity to exchange precision."""
        # TODO: Load step sizes from exchange info
        return f"{quantity:.6f}"

    def _format_price(self, symbol: str, price: float) -> str:
        """Format price to exchange precision."""
        # TODO: Load tick sizes from exchange info
        return f"{price:.8f}"

    def _check_rate_limit(self):
        """Enforce request rate limits."""
        now = time.time()
        window = 60.0
        self._request_timestamps = [t for t in self._request_timestamps if now - t < window]
        if len(self._request_timestamps) >= self.config.requests_per_minute:
            sleep_time = self._request_timestamps[0] + window - now
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._request_timestamps.append(now)

    def _check_order_rate_limit(self):
        """Enforce order-specific rate limits."""
        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 1.0]
        if len(self._order_timestamps) >= self.config.orders_per_second:
            time.sleep(0.1)
        self._order_timestamps.append(now)

    def close(self):
        """Close the HTTP client."""
        self._client.close()
