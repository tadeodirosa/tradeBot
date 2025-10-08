#!/usr/bin/env python3
"""
🔍 EXPLORADOR DE NUEVAS COINS
Descargar y analizar nuevas cryptocurrencies para encontrar mejores oportunidades
"""

import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# Lista de nuevas coins a explorar
NEW_COINS = [
    'XRP',     # Ripple - muy popular
    'AVAX',    # Avalanche - Layer 1
    'LINK',    # Chainlink - Oracle
    'DOT',     # Polkadot - Interoperability
    'UNI',     # Uniswap - DEX
    'LTC',     # Litecoin - Classic
    'BCH',     # Bitcoin Cash
    'ATOM',    # Cosmos - Ecosystem
    'ICP',     # Internet Computer
    'FIL',     # Filecoin - Storage
    'TRX',     # Tron
    'NEAR',    # Near Protocol
    'ALGO',    # Algorand
    'VET',     # VeChain
    'THETA',   # Theta Network
]

class CoinDataDownloader:
    """Descargador de datos para nuevas coins"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache_real')
        
        # Crear directorio si no existe
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_available_symbols(self):
        """Obtener símbolos disponibles en Binance"""
        try:
            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            symbols = []
            
            for symbol_info in data['symbols']:
                symbol = symbol_info['symbol']
                status = symbol_info['status']
                
                # Solo futuros USDT activos
                if symbol.endswith('USDT') and status == 'TRADING':
                    base_asset = symbol.replace('USDT', '')
                    if base_asset in NEW_COINS:
                        symbols.append(symbol)
            
            return symbols
            
        except Exception as e:
            print(f"❌ Error obteniendo símbolos: {e}")
            return []
    
    def download_klines(self, symbol, interval='4h', limit=1000):
        """Descargar datos OHLCV"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            print(f"📥 Descargando {symbol} {interval}...")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convertir a formato estándar
            formatted_data = []
            for kline in data:
                formatted_data.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            return formatted_data
            
        except Exception as e:
            print(f"❌ Error descargando {symbol}: {e}")
            return None
    
    def save_data(self, symbol, interval, data):
        """Guardar datos en cache"""
        try:
            filename = f"{symbol}_{interval}.json"
            filepath = os.path.join(self.cache_dir, filename)
            
            cache_data = {
                'symbol': symbol,
                'interval': interval,
                'last_update': datetime.now().isoformat(),
                'count': len(data),
                'data': data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Guardado: {filename} ({len(data)} barras)")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando {symbol}: {e}")
            return False
    
    def download_all_new_coins(self):
        """Descargar todas las nuevas coins disponibles"""
        print("🔍 EXPLORANDO NUEVAS COINS")
        print("=" * 50)
        
        # Obtener símbolos disponibles
        available_symbols = self.get_available_symbols()
        
        if not available_symbols:
            print("❌ No se pudieron obtener símbolos")
            return
        
        print(f"✅ Encontrados {len(available_symbols)} nuevos símbolos:")
        for symbol in available_symbols:
            print(f"   • {symbol}")
        print()
        
        # Descargar datos para cada símbolo y timeframe
        timeframes = ['1h', '4h', '1d']
        successful_downloads = []
        
        for symbol in available_symbols:
            coin_success = True
            
            for timeframe in timeframes:
                # Esperar entre requests para evitar rate limit
                time.sleep(0.5)
                
                data = self.download_klines(symbol, timeframe)
                
                if data:
                    if self.save_data(symbol, timeframe, data):
                        successful_downloads.append(f"{symbol}_{timeframe}")
                    else:
                        coin_success = False
                else:
                    coin_success = False
            
            if coin_success:
                print(f"✅ {symbol} completado")
            else:
                print(f"⚠️ {symbol} parcialmente completado")
            
            # Pausa entre coins
            time.sleep(1)
        
        print(f"\n🎯 RESUMEN:")
        print(f"   Total descargas exitosas: {len(successful_downloads)}")
        print(f"   Nuevas coins disponibles: {len(available_symbols)}")
        
        return successful_downloads

def analyze_new_coin_performance():
    """Analizar rápidamente las nuevas coins"""
    print("\n📊 ANÁLISIS RÁPIDO DE NUEVAS COINS")
    print("=" * 50)
    
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache_real')
    results = []
    
    # Buscar archivos 4h de nuevas coins
    for coin in NEW_COINS:
        filename = f"{coin}USDT_4h.json"
        filepath = os.path.join(cache_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                
                data = cache['data']
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Análisis básico de últimos 30 días
                recent_data = df.tail(180)  # ~30 días en 4h
                
                if len(recent_data) >= 10:
                    first_price = recent_data['close'].iloc[0]
                    last_price = recent_data['close'].iloc[-1]
                    
                    returns_30d = ((last_price - first_price) / first_price) * 100
                    volatility = recent_data['close'].pct_change().std() * 100
                    avg_volume = recent_data['volume'].mean()
                    
                    # Score simple de oportunidad
                    opportunity_score = 50
                    
                    # Volatilidad moderada es buena
                    if 2 < volatility < 8:
                        opportunity_score += 15
                    elif volatility > 15:
                        opportunity_score -= 10
                    
                    # Volumen decente
                    if avg_volume > 1000000:  # 1M+
                        opportunity_score += 10
                    
                    # Tendencia reciente
                    if returns_30d > 5:
                        opportunity_score += 10
                    elif returns_30d < -20:
                        opportunity_score += 5  # Oportunidad de rebote
                    
                    results.append({
                        'coin': coin,
                        'returns_30d': returns_30d,
                        'volatility': volatility,
                        'avg_volume': avg_volume,
                        'opportunity_score': opportunity_score,
                        'current_price': last_price
                    })
                    
            except Exception as e:
                print(f"⚠️ Error analizando {coin}: {e}")
    
    # Ordenar por score de oportunidad
    results.sort(key=lambda x: x['opportunity_score'], reverse=True)
    
    print("🏆 TOP NUEVAS COINS POR OPORTUNIDAD:")
    print("-" * 70)
    print(f"{'Coin':<8} {'Score':<6} {'Ret30d':<8} {'Volat':<7} {'Precio':<12} {'Volumen'}")
    print("-" * 70)
    
    for result in results[:10]:  # Top 10
        print(f"{result['coin']:<8} "
              f"{result['opportunity_score']:<6.0f} "
              f"{result['returns_30d']:>+6.1f}% "
              f"{result['volatility']:>5.1f}% "
              f"${result['current_price']:>8.4f} "
              f"{result['avg_volume']:>10,.0f}")
    
    return results

def main():
    """Función principal"""
    print("🚀 EXPLORADOR DE NUEVAS CRYPTOCURRENCIES")
    print("Buscando mejores oportunidades que ETH/SOL/BTC...")
    print()
    
    # Descargar nuevas coins
    downloader = CoinDataDownloader()
    downloads = downloader.download_all_new_coins()
    
    if downloads:
        # Analizar performance
        results = analyze_new_coin_performance()
        
        if results:
            print(f"\n✅ Análisis completado. {len(results)} nuevas coins disponibles.")
            print("\n🎯 RECOMENDACIONES:")
            
            top_3 = results[:3]
            for i, coin_data in enumerate(top_3, 1):
                coin = coin_data['coin']
                score = coin_data['opportunity_score']
                print(f"   {i}. {coin}USDT (Score: {score:.0f}) - ¡PROBAR BACKTEST!")
        else:
            print("❌ No se pudieron analizar las nuevas coins")
    else:
        print("❌ No se pudieron descargar nuevas coins")

if __name__ == "__main__":
    main()