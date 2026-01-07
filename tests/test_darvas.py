import unittest
import numpy as np
from src.strategy import TradingStrategy


class TestDarvasBoxStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = TradingStrategy(initial_capital=10000)

    def test_flat_line(self):
        # Flat line, no boxes should form or trigger trades ideally, or just stay in state
        prices = np.array([100.0] * 20)
        signals, portfolio = self.strategy.darvas_box_strategy(prices, prices)
        # Should be no signals (0)
        self.assertTrue(all(s == 0 for s in signals))

    def test_box_formation_and_breakout(self):
        # 1. Identify Top: 100, 99, 99, 99 (at i=3, top=100 confirmed)
        # 2. Identify Bottom: 90, 91, 91, 91 (at i=7, bottom=90 confirmed)
        # 3. Breakout: 101 -> Buy

        prices = [
            100,  # 0: Potential High
            95,  # 1
            95,  # 2
            95,  # 3: Top confirmed (High 100). State -> TOP_SET. Low starts tracking (95)
            90,  # 4: Potential Low
            92,  # 5
            92,  # 6
            92,  # 7: Bottom confirmed (Low 90). State -> BOX_ESTABLISHED
            95,  # 8
            102,  # 9: Breakout! (102 > 100). Buy Signal.
            105,  # 10
        ]
        prices = np.array(prices, dtype=float)
        signals, _ = self.strategy.darvas_box_strategy(prices, prices)

        # We expect a buy signal at index 9
        # Python lists are 0-indexed.
        # Let's check the signals array.
        # signals[9] should be 1

        print("Signals:", signals)
        self.assertEqual(signals[9], 1)

    def test_stop_loss(self):
        # Buy then drop
        prices = [100, 95, 95, 95, 80, 85, 85, 85, 90, 105, 100, 90, 80]  # Buy at 105  # Drop
        prices = np.array(prices, dtype=float)

        # Stop loss 10%
        # Entry 105. SL = 94.5.
        # Price drops to 90 -> Trigger sell

        signals, _ = self.strategy.darvas_box_strategy(prices, prices, stop_loss_pct=0.10)

        # Find index of 105 => 9
        # Index 10=100
        # Index 11=90 -> Sold

        self.assertEqual(signals[11], -1)


if __name__ == "__main__":
    unittest.main()
