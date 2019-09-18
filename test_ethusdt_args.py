# imports
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import pandas as pd

import datetime  # For datetime objects
import pprint
import itertools

import backtrader as bt

from binance.client import Client
from klychi import api_key, api_secret

client = Client(api_key, api_secret)


def organise_binance_klines(df, save_csv=False, csv_name=''):
    col_names = ['open_time_unix', 'open', 'high', 'low', 'close', 'volume',
                 'close_time_unix', 'quote_asset_volume', 'number_of_trades',
                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    df_new = df.copy()

    # change unix time to datetime; binance show time in ms
    df_new['candle-start'] = pd.to_datetime(df_new[0], unit='ms')
    df_new['candle-end'] = pd.to_datetime(df_new[6], unit='ms')

    # rename columns
    df_new.columns = col_names + ['candle-start', 'candle-end']

    # set index and drop extra columns
    df_new.set_index(['candle-end'], inplace=True)

    # change dtypes of data from object to float/int
    df_new = df_new.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64',
                            'volume': 'float64', 'quote_asset_volume': 'float64', 'number_of_trades': 'int64',
                            'taker_buy_base_asset_volume': 'float64',
                            'taker_buy_quote_asset_volume': 'float64'})

    if save_csv:
        df_new.to_csv(path_or_buf=str(csv_name) + '.csv')

    return df_new


def printTradeAnalysis(analyzer):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    # Get the results we are interested in
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total, 2)
    strike_rate = round(number=((total_won / total_closed) * 100), ndigits=2)
    # Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate %', 'Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed, total_won, total_lost]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net]
    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('', *row))


def printDrawDown(analyzer):
    '''
    Function to print the DrawDown results in a nice format.
    '''
    ddpct = round(analyzer.drawdown, 2)
    moneydd = round(analyzer.moneydown, 2)
    ddlen = analyzer.len
    maxdd = round(analyzer.max.drawdown, 2)
    maxmoneydd = round(analyzer.max.moneydown, 2)
    maxddlen = analyzer.max.len

    # Designate the rows
    h1 = ['DrawDown %', 'Money Down $', 'DrawDown Length']
    h2 = ['Max DrawDown %', 'Max MoneyDown $ ', 'Max DrawDown Length']
    r1 = [ddpct, moneydd, ddlen]
    r2 = [maxdd, maxmoneydd, maxddlen]

    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("DrawDown Results:")
    for row in print_list:
        print(row_format.format('', *row))


class TestStrategy(bt.Strategy):
    params = (
        ('short_ema', 17),
        ('long_ema', 48),
        ('mult', 10),  #  multiply profits and loses (margin trading)
        ('lever', 10),
        ('printlog', True),
        ('trailstoppct', 0.05),  # trailstop order trail 0.05 = 5%
    )


    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add exponential moving averages
        ema_short = bt.ind.EMA(period=self.params.short_ema)
        ema_long = bt.ind.EMA(period=self.params.long_ema)

        self.crossover = bt.ind.CrossOver(ema_short, ema_long)
        self.pricecrossover = bt.ind.CrossOver(self.datas[0].close, ema_long)

        self.wait_to_enter = ''


    def next(self):

        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # if an order is active, no new orders are allowed
        if self.order:
            return

            # if position is not opened
        if not self.position:

            # if signal is go long - cross upwards
            if self.crossover > 0:

                # if we are waiting for a Short signal
                if self.wait_to_enter == 'wait_for_short':
                    return

                else:  # BUY BUY BUY
                    self.log('BUY CREATE , %.2f' % self.dataclose[0])
                    self.wait_to_enter = ''
                    # to make orders at 5% of portfolio value, no more than 500$
                    self.order = self.buy(size=round(min((cerebro.broker.getvalue() * 0.05),
                                                         500) / self.dataclose[0], 2))
                    # exectype=bt.Order.StopTrail, trailpercent=self.params.trailstoppct)

            # if signal is go short - cross downwards
            elif self.crossover < 0:

                # if we are waiting for a Long signal
                if self.wait_to_enter == 'wait_for_long':
                    return

                else:  # SELL SELL SELL
                    self.log('SELL CREATE , %.2f' % self.dataclose[0])
                    self.wait_to_enter = ''
                    # to make orders at 5% of portfolio value, no more than 500$
                    self.order = self.sell(size=round(min((cerebro.broker.getvalue() * 0.05),
                                                          500) / self.dataclose[0], 2))
                    # exectype=bt.Order.StopTrail, trailpercent=self.params.trailstoppct)

        # if we are in the market:
        elif self.position:
            # if price signals close short == -1
            if self.pricecrossover > 0:
                self.log('CLOSE SHORT , %.2f , Wait for Long' % self.dataclose[0])
                self.wait_to_enter = 'wait_for_long'
                self.order = self.close()

            # if price signals close long == 1
            elif self.pricecrossover < 0:
                self.log('CLOSE LONG , %.2f , Wait for Short' % self.dataclose[0])
                self.wait_to_enter = 'wait_for_short'
                self.order = self.close()


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:

            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm)
                         )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm

            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm)
                         )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected', doprint=True)

        # Write down: no pending order
        self.order = None


    def notify_trade(self, trade):
        if trade.isopen:
            self.log('Trade id: {}'.format(trade.ref))

        elif trade.isclosed:
            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f, Acc Balance: %.2f' %
                     (trade.pnl, trade.pnlcomm, cerebro.broker.getvalue()))

        elif trade.justopened:
            self.log('TRADE OPENED, SIZE: %2d , VAL: %.2f' % (trade.size, trade.value))


def runstrat(args=None):

    args = parse_args(args)
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    eth_usdt1h = pd.read_csv(args.data, index_col='candle-end')

    eth_usdt1h.index = pd.to_datetime(eth_usdt1h.index)
    eth_usdt1h['candle-start'] = pd.to_datetime(eth_usdt1h['candle-start'])

    # timeframe and compression tells the system we have hourly data
    data = bt.feeds.PandasData(dataname=eth_usdt1h,
                               timeframe=bt.TimeFrame.Minutes,
                               compression=60
                               )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(args.cash)  # 10,000

    # Set the commission
    cerebro.broker.setcommission(commission=0.001,  # 0.1% binance commission
                                 mult=TestStrategy.params.mult,
                                 interest=0.01,  # long/short interest 0.01 -> 1%
                                 leverage=TestStrategy.params.lever,
                                 interest_long=True
                                 )

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Create Analyzers
    # RF = 1%,
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='mysharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='mytradeanal')
    cerebro.addanalyzer(bt.analyzers.BasicTradeStats, _name='tradestats')

    # Create Observers
    cerebro.addobserver(bt.observers.DrawDown)

    # Run over everything
    thestrats = cerebro.run()
    thestrat = thestrats[0]

    # Print Analyzers
    print()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print()
    print('Annualised SharpeR:', thestrat.analyzers.mysharpe.get_analysis()['sharperatio'])
    print()
    # CAGR = (end bal / beg bal) ^ 1/n of years - 1
    print('CAGR % ', thestrat.analyzers.returns.get_analysis()['rnorm100'] * 100)
    print()
    printTradeAnalysis(thestrat.analyzers.mytradeanal.get_analysis())
    print()
    printDrawDown(thestrat.analyzers.mydrawdown.get_analysis())
    print()
    pprint.pprint(dict(thestrat.analyzers.tradestats.get_analysis()))

    # to plot part of the data
    # cerebro.plot(start=datetime.date(2017, 12, 1), end=datetime.date(2018, 1, 1),
    #              plotname='ETH/USDT 1h EMA and Price Crossover, 2017_08-2019_08',
    #              savefig=True)

    cerebro.plot()  # plot backtest


def parse_args(pargs=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Signal concepts')

    parser.add_argument('--data', required=False,
                        default='eth_usdt1h.csv',
                        help='Specific data to be read in')

    parser.add_argument('--fromdate', required=False, default=None,
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', required=False, default=None,
                        help='Ending date in YYYY-MM-DD format')

    parser.add_argument('--cash', required=False, action='store',
                        type=float, default=10000,
                        help=('Cash to start with'))

    parser.add_argument('--smaperiod', required=False, action='store',
                        type=int, default=30,
                        help=('Period for the moving average'))

    parser.add_argument('--exitperiod', required=False, action='store',
                        type=int, default=5,
                        help=('Period for the exit control SMA'))



    if pargs is not None:
        return parser.parse_args(pargs)

    return parser.parse_args()

if __name__ == "__main__":
    runstrat()