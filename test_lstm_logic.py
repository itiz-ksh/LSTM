import unittest
from pathlib import Path
import LSTM
class LSTMLogicTests(unittest.TestCase):
    def test_load_close_prices_uses_project_dataset_and_sorts_oldest_first(self):
        frame = LSTM.load_close_prices(Path("Dataset/AAPL.csv"))
        self.assertEqual(list(frame.columns), ["Date", "Close/Last"])
        self.assertGreater(len(frame), 60)
        self.assertLess(frame["Date"].iloc[0], frame["Date"].iloc[-1])
        self.assertEqual(float(frame["Close/Last"].iloc[0]), 29.8557)
        self.assertEqual(float(frame["Close/Last"].iloc[-1]), 273.36)
    def test_create_sequences_uses_previous_steps_to_predict_next_value(self):
        data = LSTM.np.array([[0.1], [0.2], [0.3], [0.4]])
        X, y = LSTM.create_sequences(data, time_steps=2)
        self.assertEqual(X.tolist(), [[0.1, 0.2], [0.2, 0.3]])
        self.assertEqual(y.tolist(), [0.3, 0.4])
if __name__ == "__main__":
    unittest.main()
